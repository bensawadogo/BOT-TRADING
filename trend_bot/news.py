"""Annonces économiques : pas de nouvelle exposition autour d'une annonce à fort impact.

Autour d'une annonce (emploi américain, taux de la banque centrale…) le spread s'élargit et
le prix saute : un ordre passé à ce moment-là coûte plus cher que prévu. Le filtre REPORTE
l'ordre (le cycle reprend 30 min plus tard) ; il ne change jamais la décision : les poids
du backtest et du live restent identiques. Une réduction ou une fermeture n'est jamais
retardée (le risque passe avant le coût).

Calendrier : flux JSON gratuit de la semaine en cours (ForexFactory via faireconomy).
Si le calendrier est injoignable, le bot garde l'ancien relevé, sinon il trade sans filtre
(avertissement) : un bot qui ne passe plus d'ordre parce qu'un site est en panne est pire.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)

URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
AVANT_MIN, APRES_MIN = 30, 30            # fenêtre autour de l'annonce
MAX_AGE_H = 6                            # relevé rafraîchi au plus toutes les 6 h

# Marchés qui ne s'écrivent pas « BASEQUOTE » : devise qui les fait bouger.
AUTRES = {"NAS100": ["USD"], "SPX500": ["USD"], "US30": ["USD"], "XAUUSD": ["USD"],
          "XAGUSD": ["USD"], "BTCUSD": ["USD"], "ETHUSD": ["USD"], "USOIL": ["USD"],
          "UKOIL": ["USD"], "NGAS": ["USD"], "GER40": ["EUR"], "UK100": ["GBP"],
          "JPN225": ["JPY"]}


def devises(marche: str) -> list[str]:
    """Devises dont les annonces touchent ce marché (EURUSD → EUR, USD)."""
    if marche in AUTRES:
        return AUTRES[marche]
    return [marche[:3], marche[3:6]] if len(marche) == 6 and marche.isalpha() else []


def _fetch() -> list[dict]:
    req = urllib.request.Request(URL, headers={"User-Agent": "trend_bot"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def charger(cache: str | Path, fetch=_fetch, maintenant: float | None = None) -> list[dict]:
    """Annonces de la semaine (relevé gardé dans `cache`, rafraîchi toutes les 6 h)."""
    cache, now = Path(cache), maintenant if maintenant is not None else time.time()
    if cache.exists() and now - cache.stat().st_mtime < MAX_AGE_H * 3600:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        events = fetch()
    except (OSError, ValueError) as e:
        log.warning("calendrier économique injoignable (%s) : %s", e,
                    "ancien relevé utilisé" if cache.exists() else "aucun filtre d'annonces")
        return json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(events), encoding="utf-8")
    return events


def bloquantes(marche: str, maintenant: datetime, events: list[dict],
               avant: int = AVANT_MIN, apres: int = APRES_MIN) -> list[str]:
    """Annonces à fort impact sur les devises du marché, dans la fenêtre [-avant ; +apres]
    minutes. Renvoie leur description (« 12:30 UTC USD Non-Farm Employment Change »)."""
    mes = set(devises(marche))
    out = []
    for e in events:
        if e.get("impact") != "High" or e.get("country") not in mes:
            continue
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except (KeyError, ValueError):
            continue
        if t - timedelta(minutes=avant) <= maintenant <= t + timedelta(minutes=apres):
            out.append(f"{t:%H:%M} UTC {e['country']} {e.get('title', '')}".strip())
    return out
