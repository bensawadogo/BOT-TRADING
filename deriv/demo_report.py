"""
Verdict de la démo : la stratégie drift est-elle rentable en conditions réelles ?

Lit le journal SQLite (trades `spike_drift_binary` clôturés) et applique le
protocole décidé dans TASKS.md, avec le seuil de rentabilité calculé sur les
VRAIS payouts Deriv (pas une constante) :

    seuil = 1 / (1 + payout moyen des trades gagnés)

Passage en réel seulement si : ≥ 200 trades ET p < 0.05 ET borne basse de
Wilson (95 %) > seuil. La décision reste humaine (DERIV_ALLOW_REAL).

Les trades au résultat INCONNU (coupure pendant le règlement) sont exclus des
statistiques — les compter perdus fausserait le verdict — mais signalés, avec
leur contract_id pour vérification manuelle sur Deriv.

Usage : python deriv/demo_report.py [--db chemin.db] [--symbol BOOM500]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from math import sqrt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MIN_TRADES = 200
PHASE = "spike_drift_binary"
BREAKEVEN_DEFAUT = 0.556   # payout 80 % tant qu'aucun trade gagné n'est connu


def wilson(n: int, wins: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    marge = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - marge, centre + marge)


def p_value(n: int, wins: int, p0: float) -> float:
    """H0 : taux de réussite <= seuil de rentabilité."""
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(wins, n, p0, alternative="greater").pvalue)


def charger_trades(db_path: str, symbol: str | None = None) -> list[tuple]:
    """(pnl_dollar, stake, notes) ; pnl_dollar None = trade jamais clôturé."""
    sql = ("SELECT pnl_dollar, stake, COALESCE(notes, '') FROM trades "
           "WHERE strategy_phase = ?")
    args: list = [PHASE]
    if symbol:
        sql += " AND symbol = ?"
        args.append(symbol)
    with sqlite3.connect(db_path) as conn:
        return conn.execute(sql, args).fetchall()


def _inconnu(note: str) -> bool:
    return "inconnu" in note


def verdict(trades: list[tuple]) -> dict:
    inconnus = [t for t in trades if t[0] is not None and _inconnu(t[2])]
    non_clotures = [t for t in trades if t[0] is None]
    trades = [t for t in trades if t[0] is not None and not _inconnu(t[2])]
    n = len(trades)
    gains = [(p, s) for p, s, _ in trades if p > 0]
    wins = len(gains)
    payouts = [p / s for p, s in gains if s]
    payout = sum(payouts) / len(payouts) if payouts else None
    seuil = 1 / (1 + payout) if payout else BREAKEVEN_DEFAUT
    lo, hi = wilson(n, wins)
    pv = p_value(n, wins, seuil)
    wr = wins / n if n else 0.0

    if n >= MIN_TRADES and pv < 0.05 and lo > seuil:
        decision = "EDGE CONFIRMÉ : passage en réel envisageable (petite mise)"
    elif n >= MIN_TRADES and hi < seuil:
        decision = "PAS D'EDGE : arrêter la stratégie"
    elif n >= MIN_TRADES:
        decision = "NON CONCLUANT : rester en démo et continuer à collecter"
    else:
        decision = f"EN COURS : {n}/{MIN_TRADES} trades, rester en démo"

    return {
        "n_trades": n, "n_wins": wins, "win_rate": wr,
        "payout_moyen": payout, "seuil_rentabilite": seuil,
        "wilson_lo": lo, "wilson_hi": hi, "p_value": pv,
        "pnl_total": sum(p for p, _, _ in trades),
        "resultats_inconnus": len(inconnus),
        "contrats_a_verifier": [note for *_, note in inconnus],
        "non_clotures": len(non_clotures),
        "decision": decision,
    }


def main(argv: list[str] | None = None) -> dict:
    from deriv.trading_journal import DB_PATH

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--symbol", default=None)
    args = ap.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"Journal introuvable : {args.db}")
        return {}
    r = verdict(charger_trades(args.db, args.symbol))
    payout = f"{r['payout_moyen']:.1%}" if r["payout_moyen"] else "inconnu"
    print("=" * 60)
    print(f"DÉMO STRATÉGIE DRIFT {args.symbol or ''}".strip())
    print("=" * 60)
    print(f"Trades comptés      : {r['n_trades']}")
    if r["resultats_inconnus"] or r["non_clotures"]:
        print(f"Exclus              : {r['resultats_inconnus']} résultat(s) inconnu(s), "
              f"{r['non_clotures']} non clôturé(s) — à vérifier sur Deriv :")
        for note in r["contrats_a_verifier"]:
            print(f"   - {note}")
    print(f"Taux de réussite    : {r['win_rate']:.1%}  [Wilson 95 % : "
          f"{r['wilson_lo']:.1%} – {r['wilson_hi']:.1%}]")
    print(f"Payout moyen        : {payout}  → seuil de rentabilité {r['seuil_rentabilite']:.1%}")
    print(f"p-value (vs seuil)  : {r['p_value']:.4f}")
    print(f"P&L total           : {r['pnl_total']:+.2f} $")
    print("-" * 60)
    print(r["decision"])
    return r


if __name__ == "__main__":
    main()
