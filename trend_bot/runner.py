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

from trend_bot import config, news
from trend_bot.portfolio import needs_trade, target_weights, weight_to_lots
from trend_bot.risk import KillSwitch, apply_regime, regime_now
from trend_bot.signals import daily_returns, ex_ante_vol, market_signal, per_market, weekdays_only

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
    signal_brut: float = float("nan")   # tendance avant le filtre de régime
    exposition: float = 1.0             # multiplicateur du coupe-circuit (1, 0,5 ou 0)
    raison: str = ""                    # pourquoi cette décision, en clair


REGIMES = {1: "haussier", 0: "neutre", -1: "baissier"}


def raison(sig_brut: float, regime: int | None, sig: float, vol: float, poids: float,
           mult: float, cible: float, actuel: float, ordre: bool, facteur: float = 0.0) -> str:
    """Chaîne de décision d'un marché, lisible : chaque ordre (ou absence d'ordre) est expliqué."""
    if pd.isna(sig_brut) or pd.isna(vol):
        return "historique insuffisant : aucun signal, aucune position"
    p = [f"tendance {sig_brut:+.2f}"]
    if regime is None:
        p.append("régime inconnu (historique court) : non filtré")
    elif sig != sig_brut:
        p.append(f"régime {REGIMES[regime]} contredit la tendance → signal {sig:+.2f}"
                 + (" (coupé)" if sig == 0 else f" (×{facteur:g})"))
    else:
        p.append(f"régime {REGIMES[regime]} : pas de contradiction")
    p.append(f"vol {vol:.1%} → poids {poids:+.3f}")
    if mult != 1:
        p.append("coupe-circuit : tout est coupé" if mult == 0
                 else f"coupe-circuit : exposition ×{mult:g}")
    if not ordre and cible != actuel:
        p.append("zone neutre : écart trop petit, pas d'ordre (frais évités)")
    elif not ordre:
        p.append("position déjà conforme")
    return " ; ".join(p)


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, st: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st, indent=2))


def _journal(path: Path, lignes: list[Ligne]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(Ligne.__dataclass_fields__)
    entete = (path.read_text(encoding="utf-8").splitlines() or [""])[0] if path.exists() else ""
    if entete and entete.split(",") != cols:
        path.replace(path.with_name(f"{path.stem}_ancien_format{path.suffix}"))   # colonnes changées
    neuf = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if neuf:
            w.writeheader()
        for l in lignes:
            w.writerow(asdict(l))


def equity_path(journal_path: str | Path) -> Path:
    """Courbe d'équité (une ligne par cycle), à côté du journal des décisions."""
    return Path(journal_path).with_name("trend_equity.csv")


def _equity(path: Path, jour: str, equity: float, mult: float, levier: float, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    neuf = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if neuf:
            w.writerow(["horodatage_utc", "jour", "equite", "exposition", "levier_brut", "mode"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), jour,
                    round(equity, 2), mult, round(levier, 3), mode])


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
              marches: list[str] | None = None,
              maintenant: datetime | None = None) -> list[Ligne]:
    state_path, journal_path = Path(state_path), Path(journal_path)
    marches = marches or config.symbols()
    if not dry_run and not getattr(broker, "can_trade", lambda: True)():
        raise RuntimeError("trading algorithmique désactivé dans MT5 : clique sur « Algo Trading » "
                           "(Ctrl+E) dans le terminal, puis relance. Aucun ordre n'a été envoyé.")
    maintenant = maintenant or datetime.now(timezone.utc)
    annonces = []
    if config.NEWS_FILTER and not dry_run:
        annonces = news.charger(config.NEWS_PATH)

    # 1. Données : clôtures journalières terminées
    noms, closes = {}, {}
    for m in marches:
        if m not in config.UNIVERSE:
            log.warning("%s est hors du panier par défaut (trend_bot/config.py) : "
                        "non backtesté ou swaps prohibitifs", m)
        sym = broker.resolve(m, config.ALIASES.get(m, [m]))
        if sym is None:
            log.warning("%s introuvable chez le courtier : ignoré", m)
            continue
        c = weekdays_only(broker.daily_closes(sym, config.HISTORY_DAYS))
        if len(c) < 253:
            log.warning("%s : %d jours d'historique (< 253) : ignoré", sym, len(c))
            continue
        noms[m], closes[m] = sym, c
    if not closes:
        raise RuntimeError("aucun marché exploitable")
    df = pd.DataFrame(closes).sort_index()

    # 2. Alpha + régime (mêmes séries et mêmes fonctions que trend_bot/backtest.py)
    strat = config.STRATEGIE
    sig_brut = per_market(lambda c: market_signal(c, strat), df).iloc[-1]
    vol = per_market(ex_ante_vol, df).iloc[-1]
    regimes = {m: regime_now(closes[m]) for m in closes}
    for m in [m for m, r in regimes.items() if r is None]:
        log.warning("%s : historique trop court pour le filtre de régime, position non filtrée", m)
    sig = pd.Series({m: apply_regime(sig_brut[m], regimes[m], strat.regime_facteur)
                     for m in sig_brut.index})

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
    lignes, trop_petits = [], {}
    # Positions du bot sur un marché sorti du panier (ou devenu introuvable) : fermées.
    orphelins = {s: s for s in pos if s not in noms.values()}
    for m, sym in [*noms.items(), *orphelins.items()]:
        spec = broker.spec(sym)
        poids = float(w.get(m, 0.0)) if m in noms else 0.0
        cible = weight_to_lots(poids, equity, spec)
        actuel = pos.get(sym, 0.0)
        action, res = "aucune", ""
        if poids and not cible and spec.lot_value <= 0:
            res = "prix indisponible : pas d'ordre"
        elif poids and not cible:
            trop_petits[m] = spec.volume_min * spec.lot_value / abs(poids)
            res = f"sous le lot minimum (équité nécessaire ≈ {trop_petits[m]:,.0f})"
        ordre = needs_trade(actuel, cible, spec.volume_step)
        attente = []
        # Ouvre ou augmente une exposition (y compris un retournement) : jamais pour réduire ou fermer
        ouvre = cible != 0 and (actuel == 0 or cible * actuel < 0 or abs(cible) > abs(actuel) + 1e-9)
        if ordre and annonces and m in noms and ouvre:
            attente = news.bloquantes(m, maintenant, annonces)
        if attente:
            action, res = "reporté", f"reporté : annonce {attente[0]}"
        elif ordre:
            delta = round(cible - actuel, 8)
            action = f"{'achat' if delta > 0 else 'vente'} {abs(delta):g}"
            if m not in noms:
                action += " (hors panier : fermeture)"
            if dry_run:
                res = "simulation"
            else:
                r = broker.market_order(sym, delta, "trend_bot")
                res = f"{'ok' if r.ok else 'ÉCHEC'} {r.lots:g} @ {r.price} {r.message}".strip()
        if m in noms:
            why = raison(sig_brut[m], regimes[m], sig[m], vol[m], poids, mult, cible, actuel,
                         ordre and not attente, strat.regime_facteur)
            if attente:
                why += f" ; ordre reporté de 30 min : annonce à fort impact ({attente[0]})"
            lignes.append(Ligne(jour, m, sym, round(float(sig[m]), 3), regimes[m],
                                round(float(vol[m]), 4), round(poids, 4), actuel, cible,
                                action, res, round(float(sig_brut[m]), 3), mult, why))
        else:
            lignes.append(Ligne(jour, m, sym, float("nan"), None, float("nan"), 0.0,
                                actuel, cible, action, res, float("nan"), mult,
                                "marché sorti du panier : position fermée"))
    _journal(journal_path, lignes)
    _equity(equity_path(journal_path), jour, equity, mult, float(w.abs().sum()),
            "simulation" if dry_run else "réel")
    st.update(peak=ks.peak, tripped=ks.tripped, last_run=jour, equity=equity,
              mode="simulation" if dry_run else "réel")
    _save_state(state_path, st)

    trades = [l for l in lignes if l.action != "aucune"]
    reportes = [l for l in trades if l.action == "reporté"]
    resume = (f"trend_bot {jour} ({'simulation' if dry_run else 'RÉEL'}) — équité {equity:,.2f}, "
              f"exposition ×{mult:g}, levier brut {w.abs().sum():.2f}, "
              f"{len(trades) - len(reportes)} ordre(s)"
              + (f", {len(reportes)} reporté(s) (annonce économique)" if reportes else ""))
    if trop_petits:
        resume += (f"\n⚠️ {len(trop_petits)} marché(s) sous le lot minimum, non tradés : "
                   f"{', '.join(trop_petits)} (il faudrait ≈ {max(trop_petits.values()):,.0f} "
                   "d'équité pour tous les trader)")
    log.info(resume)
    for l in trades:
        log.info("  %s %s → %s %s | %s", l.symbole, l.action, l.lots_cibles, l.resultat, l.raison)
    telegram(resume + "".join(f"\n{l.symbole}: {l.action} {l.resultat}\n   ↳ {l.raison}"
                              for l in trades))
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


def empecher_la_veille() -> bool:
    """Windows : tant que ce processus vit, le PC ne se met pas en veille automatiquement
    (SetThreadExecutionState, ce que font les lecteurs vidéo). N'empêche pas une mise en veille
    manuelle ni le couvercle d'un portable : la tâche planifiée (scripts/lancer_bot.ps1) réveille
    le PC avant 7 h UTC pour ces cas-là. Renvoie False si indisponible (autre OS)."""
    try:
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    except (AttributeError, OSError):
        return False


def loop(broker, dry_run: bool, state_path: str | Path = config.STATE_PATH) -> None:
    """Un cycle par jour en semaine entre RUN_HOUR_UTC (7 h : forex, indices et métaux ouverts,
    bougie D1 de la veille terminée) et RUN_END_UTC ; jamais deux fois le même jour. Un cycle
    dont des ordres sont reportés pour une annonce est repris 30 min plus tard."""
    log.info("veille automatique %s", "bloquée tant que le bot tourne" if empecher_la_veille()
             else "non bloquée (OS sans SetThreadExecutionState)")
    while True:
        now = datetime.now(timezone.utc)
        st = _load_state(Path(state_path))
        if (now.weekday() < 5 and config.RUN_HOUR_UTC <= now.hour < config.RUN_END_UTC
                and st.get("cycle_utc_date") != now.date().isoformat()):
            try:
                lignes = run_cycle(broker, dry_run, state_path)
            except Exception as e:      # le bot ne doit pas mourir sur une erreur réseau
                log.exception("cycle en échec")
                telegram(f"⚠️ trend_bot : cycle en échec, nouvel essai dans 30 min : {e}")
                time.sleep(1800)
                continue
            if any(l.action == "reporté" for l in lignes):
                time.sleep(1800)         # annonce en cours : on reprend, les ordres déjà passés restent
                continue
            st = _load_state(Path(state_path))
            st["cycle_utc_date"] = now.date().isoformat()
            _save_state(Path(state_path), st)
        time.sleep(300)
