"""Un cycle journalier du bot, identique au backtest (trend_bot/backtest.py) :

clôtures terminées → signal de tendance → filtre de régime → poids cibles (vol cible)
→ coupe-circuit de drawdown → lots → zone neutre → ordres (ou simulation) → journal.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trend_bot import config
from trend_bot.portfolio import needs_trade, target_weights, weight_to_lots
from trend_bot.risk import KillSwitch, apply_regime, regime_now
from trend_bot.signals import daily_returns, ex_ante_vol, per_market, trend_signal

log = logging.getLogger(__name__)


@dataclass
class Ligne:
    date: str
    marche: str
    symbole: str
    signal: float
    regime: int | None
    vol: float
    poids: float
    lots_actuels: float
    lots_cibles: float
    action: str
    resultat: str


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=2))


def _journal(path: Path, lignes: list[Ligne]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    neuf = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(Ligne.__dataclass_fields__))
        if neuf:
            w.writeheader()
        for l in lignes:
            w.writerow(asdict(l))


def telegram(texte: str) -> None:
    tok, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not (tok and chat):
        return
    data = urllib.parse.urlencode({"chat_id": chat, "text": texte}).encode()
    try:
        urllib.request.urlopen(f"https://api.telegram.org/bot{tok}/sendMessage", data, timeout=15)
    except OSError as e:
        log.warning("Telegram : %s", e)


def run_cycle(broker, dry_run: bool = True, state_path: str | Path = config.STATE_PATH,
              journal_path: str | Path = config.JOURNAL_PATH,
              marches: list[str] | None = None) -> list[Ligne]:
    state_path, journal_path = Path(state_path), Path(journal_path)
    marches = marches or config.symbols()

    # 1. Données : clôtures journalières terminées
    noms, closes = {}, {}
    for m in marches:
        sym = broker.resolve(m, config.UNIVERSE.get(m, [m]))
        if sym is None:
            log.warning("%s introuvable chez le courtier : ignoré", m)
            continue
        c = broker.daily_closes(sym, config.HISTORY_DAYS)
        if len(c) < 253:
            log.warning("%s : %d jours d'historique (< 253) : ignoré", sym, len(c))
            continue
        noms[m], closes[m] = sym, c
    if not closes:
        raise RuntimeError("aucun marché exploitable")
    df = pd.DataFrame(closes).sort_index()

    # 2. Alpha + régime
    sig = per_market(trend_signal, df).iloc[-1]
    vol = per_market(ex_ante_vol, df).iloc[-1]
    regimes = {m: regime_now(closes[m]) for m in closes}
    sig = pd.Series({m: apply_regime(sig[m], regimes[m]) for m in sig.index})

    # 3. Poids cibles + coupe-circuit
    w = target_weights(sig, vol, daily_returns(df).iloc[-300:],
                       port_vol=config.PORT_VOL, max_gross=config.MAX_GROSS)
    equity = broker.equity()
    st = _load_state(state_path)
    ks = KillSwitch(config.SOFT_DD, config.HARD_DD, st.get("peak", 0.0), st.get("tripped", False))
    mult = ks.update(equity)
    w *= mult
    if ks.tripped and not st.get("tripped"):
        telegram(f"⛔ trend_bot : drawdown ≥ {config.HARD_DD:.0%}, tout est coupé. "
                 "Remise à zéro manuelle : python main.py trend --reset")

    # 4. Lots + ordres
    pos = broker.positions()
    jour = df.index[-1].strftime("%Y-%m-%d")
    lignes = []
    for m, sym in noms.items():
        spec = broker.spec(sym)
        cible = weight_to_lots(float(w.get(m, 0.0)), equity, spec)
        actuel = pos.get(sym, 0.0)
        action, res = "aucune", ""
        if needs_trade(actuel, cible, spec.volume_step):
            delta = round(cible - actuel, 8)
            action = f"{'achat' if delta > 0 else 'vente'} {abs(delta):g}"
            if dry_run:
                res = "simulation"
            else:
                r = broker.market_order(sym, delta, "trend_bot")
                res = f"{'ok' if r.ok else 'ÉCHEC'} {r.lots:g} @ {r.price} {r.message}".strip()
        lignes.append(Ligne(jour, m, sym, round(float(sig[m]), 3), regimes[m],
                            round(float(vol[m]), 4), round(float(w.get(m, 0.0)), 4),
                            actuel, cible, action, res))
    _journal(journal_path, lignes)
    st.update(peak=ks.peak, tripped=ks.tripped, last_run=jour, equity=equity,
              mode="simulation" if dry_run else "réel")
    _save_state(state_path, st)

    trades = [l for l in lignes if l.action != "aucune"]
    resume = (f"trend_bot {jour} ({'simulation' if dry_run else 'RÉEL'}) — équité {equity:,.2f}, "
              f"exposition ×{mult:g}, levier brut {w.abs().sum():.2f}, {len(trades)} ordre(s)")
    log.info(resume)
    for l in trades:
        log.info("  %s %s → %s %s", l.symbole, l.action, l.lots_cibles, l.resultat)
    telegram(resume + "".join(f"\n{l.symbole}: {l.action} {l.resultat}" for l in trades))
    return lignes


def reset_killswitch(broker, state_path: str | Path = config.STATE_PATH) -> None:
    state_path = Path(state_path)
    st = _load_state(state_path)
    st.update(peak=broker.equity(), tripped=False)
    _save_state(state_path, st)


def check_account(broker, live: bool) -> None:
    """En réel, refuse un compte réel ou inconnu sans MT5_ALLOW_REAL=1."""
    demo = broker.is_demo()
    if live and demo is not True and not config.ALLOW_REAL:
        raise RuntimeError("compte réel (ou type inconnu) : refusé sans MT5_ALLOW_REAL=1")


def loop(broker, dry_run: bool, state_path: str | Path = config.STATE_PATH) -> None:
    """Un cycle par jour en semaine à partir de RUN_HOUR_UTC (7 h : forex, indices et métaux
    ouverts, bougie D1 de la veille terminée) ; jamais deux fois le même jour."""
    while True:
        now = datetime.now(timezone.utc)
        st = _load_state(Path(state_path))
        if (now.weekday() < 5 and now.hour >= config.RUN_HOUR_UTC
                and st.get("cycle_utc_date") != now.date().isoformat()):
            try:
                run_cycle(broker, dry_run, state_path)
            except Exception as e:      # le bot ne doit pas mourir sur une erreur réseau
                log.exception("cycle en échec")
                telegram(f"⚠️ trend_bot : cycle en échec, nouvel essai dans 30 min : {e}")
                time.sleep(1800)
                continue
            st = _load_state(Path(state_path))
            st["cycle_utc_date"] = now.date().isoformat()
            _save_state(Path(state_path), st)
        time.sleep(300)
