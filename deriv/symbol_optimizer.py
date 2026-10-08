"""
═══════════════════════════════════════════════════════════════════
DIAGNOSTIC AUTOMATIQUE WIN RATE PAR SYMBOLE × TIMEFRAME
deriv/symbol_optimizer.py
═══════════════════════════════════════════════════════════════════

Teste tous les symboles/timeframes via la Walk-Forward Optimization
(embargo anti-fuite) et retourne la combinaison la plus solide.

Critères de classement :
  1. win rate WFO moyen (meilleur modèle)
  2. stabilité (écart-type des folds < 10%)
  3. significativité statistique (p-value vs hasard)

Usage :
  python deriv/symbol_optimizer.py
  python deriv/symbol_optimizer.py --count 3000
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("SYMBOL_OPT")

# Résolution robuste de la racine (fonctionne en script ET en module)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BREAKEVEN = 0.556

SYMBOLES_A_TESTER = [
    # Vrais marchés Forex (comportement humain = patterns réels)
    ("frxEURUSD",  14400, "EURUSD H4"),     # priorité 1
    ("frxGBPUSD",  14400, "GBPUSD H4"),     # priorité 2
    ("frxUSDJPY",  14400, "USDJPY H4"),     # priorité 3
    ("frxEURGBP",  14400, "EURGBP H4"),
    ("frxEURUSD",  3600,  "EURUSD H1"),
    ("frxGBPUSD",  3600,  "GBPUSD H1"),

    # Crypto Deriv (moins prévisible mais plus de volatilité)
    ("cryBTCUSD",  14400, "BTCUSD H4"),
    ("cryETHUSD",  14400, "ETHUSD H4"),

    # Métaux (trending markets = bon pour HMM)
    ("frxXAUUSD",  14400, "XAUUSD H4"),     # Or/USD — excellent pour HMM
    ("frxXAUUSD",  3600,  "XAUUSD H1"),

    # Synthétiques exclusivement H4 (plus stable qu'en M1)
    ("R_75",       14400, "R_75 H4"),
    ("R_50",       14400, "R_50 H4"),
]


def _count_pour_granularity(granularity: int) -> int:
    """Adapte le nombre de bougies selon le timeframe.

    H1  × 1000 = ~42 jours → suffisant
    H4  × 500  = ~83 jours → suffisant
    M5  × 2000 = ~7 jours
    M1  × 3000 = ~2 jours
    """
    if granularity >= 14400:   # H4
        return 500
    elif granularity >= 3600:  # H1
        return 1000
    elif granularity >= 300:   # M5, M15
        return 2000
    else:                       # M1
        return 3000


async def tester_symbole(symbol, granularity, label, count=None):
    """WFO sur une combinaison symbole × timeframe (jamais d'exception)."""
    from deriv.data_collector import DerivDataCollector
    from deriv.walk_forward_optimizer import WalkForwardOptimizer

    # Adapte le count selon la granularité si non spécifié
    if count is None:
        count = _count_pour_granularity(granularity)

    try:
        collector = DerivDataCollector(symbol, granularity)
        await collector.connect()
        df = await collector.get_candle_history_full(target_count=count)
        await collector.close()

        min_candles = 260 if granularity >= 14400 else (400 if granularity >= 3600 else 600)
        if len(df) < min_candles:
            return {"label": label, "win_rate": 0,
                    "erreur": "Pas assez de donnees"}

        # WFO adaptatif selon la granularité
        wfo = WalkForwardOptimizer(granularity=granularity)
        summary = wfo.run(df)

        if not summary:
            return {"label": label, "win_rate": 0,
                    "erreur": "Aucun resultat"}

        # Prend le meilleur modèle (avec détails pour le classement)
        best_name, best = max(
            summary.items(), key=lambda kv: kv[1]["win_rate_moyen"]
        )

        return {
            "label":       label,
            "symbol":      symbol,
            "granularity": granularity,
            "win_rate":    round(best["win_rate_moyen"], 4),
            "win_rate_std": best["win_rate_std"],
            "n_folds":     best["n_folds"],
            "n_trades":    best["n_trades_total"],
            "meilleur_modele": best_name,
            "profitable":  best["win_rate_moyen"] >= BREAKEVEN,
        }
    except Exception as e:
        return {"label": label, "win_rate": 0, "erreur": str(e)}


def classer_resultats(resultats: list[dict]) -> list[dict]:
    """
    Classement pur (testable) :
      erreurs en dernier, puis win rate desc, puis stabilité (σ asc).
    """
    def cle(r):
        erreur = 1 if "erreur" in r else 0
        return (
            erreur,
            -float(r.get("win_rate", 0.0)),
            float(r.get("win_rate_std", 1.0)),
        )

    return sorted(resultats, key=cle)


def _configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


async def main():
    import argparse

    _configure_console_utf8()
    p = argparse.ArgumentParser(description="Diagnostic symbole × timeframe")
    p.add_argument("--count", type=int, default=None,
                   help="Nombre de bougies (par défaut adapté au timeframe)")
    args = p.parse_args()

    count_desc = f"{args.count} bougies chacune" if args.count else "bougies adaptatives selon le timeframe"
    print("=== OPTIMISATION SYMBOLE x TIMEFRAME ===")
    print(f"Test de {len(SYMBOLES_A_TESTER)} combinaisons "
          f"({count_desc})...\n")

    resultats = []
    for symbol, gran, label in SYMBOLES_A_TESTER:
        print(f"Test : {label}...")
        res = await tester_symbole(symbol, gran, label, count=args.count)
        resultats.append(res)
        if "erreur" in res:
            print(f"  ERREUR : {res['erreur']}")
        else:
            status = "OK PROFITABLE" if res.get("profitable") else "sous seuil"
            print(f"  {status} | Win Rate : {res['win_rate']:.1%} "
                  f"({res.get('meilleur_modele')}, "
                  f"σ={res.get('win_rate_std', 0):.1%})")

    # Tri par win rate décroissant (erreurs en dernier)
    resultats = classer_resultats(resultats)

    print("\n=== CLASSEMENT FINAL ===")
    for i, r in enumerate(resultats, 1):
        if "erreur" in r:
            print(f"{i:2d}. {r['label']:<25} ERREUR : {r['erreur']}")
            continue
        status = "OK PROFITABLE" if r.get("profitable") else "sous seuil"
        print(f"{i:2d}. {r['label']:<25} {status:<14} "
              f"WR={r['win_rate']:.1%} σ={r['win_rate_std']:.1%} "
              f"({r['n_trades']} trades / {r['n_folds']} folds)")

    best = resultats[0] if resultats else {}
    print("\nMeilleure combinaison : "
          f"{best.get('label', 'N/A')} ({best.get('win_rate', 0):.1%})")
    print(f"   Configure dans .env : DERIV_SYMBOL={best.get('symbol', 'R_75')}")
    print(f"                         DERIV_GRANULARITY="
          f"{best.get('granularity', 60)}")

    # Export du classement (audit / suivi dans TASKS.md)
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "classement_symbols.csv")
    pd.DataFrame(resultats).to_csv(out_csv, index=False)
    print(f"\nClassement sauvegarde : {out_csv}")


if __name__ == "__main__":
    asyncio.run(main())
