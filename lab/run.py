"""
Lance tous les algorithmes sur un fichier de bougies et compare au hasard.

Usage : python -m lab.run data/frxXAUUSD_3600.csv frxXAUUSD [--h4]
Coûts : commissions Multipliers Deriv relevées le 08/10/2026 (fraction du notionnel).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from lab.engine import random_baseline, run
from lab.regime import causal_regimes
from lab.strategies import STRATEGIES

SIMS = 100
COUTS = {"frxXAUUSD": 0.16 / 500, "frxEURUSD": 0.10 / 500, "R_75": 0.29 / 1000,
         # Multipliers ×50, mise 10 $ (notionnel 500 $), relevés le 08/10/2026
         "frxGBPUSD": 0.10 / 500, "frxUSDJPY": 0.14 / 500, "frxAUDUSD": 0.20 / 500,
         "frxUSDCAD": 0.11 / 500, "frxUSDCHF": 0.15 / 500, "frxEURJPY": 0.13 / 500,
         "frxGBPJPY": 0.12 / 500}


def charger(path: str, h4: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if h4:
        df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last"}).dropna()
    return df


def evaluer(df, sym, noms, minutes):
    cost = COUTS[sym]
    t0 = time.time()
    regimes = causal_regimes(df, train_len=min(1500, len(df) // 3))
    print(f"  (régimes HMM causaux : {time.time() - t0:.0f}s)")
    split = int(len(df) * 0.7)
    lignes = []
    for nom in noms:
        sig, rules = STRATEGIES[nom](df, regimes=regimes, minutes_par_bougie=minutes)
        for periode, (a, b) in {"réglage 70%": (0, split), "test 30%": (split, len(df))}.items():
            st = run(df, sig, rules, cost, a, b).stats()
            if st["n"] == 0:
                lignes.append((nom, periode, st, None))
                continue
            base = random_baseline(df, sig, rules, cost, a, b, n_sims=SIMS)
            p = float((base >= st["moy_bps"] / 1e4).mean()) if base.size else None
            lignes.append((nom, periode, st, (p, float(base.mean() * 1e4))))
    print(f"{'algo':4} {'période':12} {'trades':>6} {'réussite':>8} {'moy/trade':>10} "
          f"{'PF':>5} {'total':>8} {'DD max':>7} | {'hasard moy':>10} {'p(hasard≥)':>10}")
    for nom, periode, st, b in lignes:
        if st["n"] == 0:
            print(f"{nom:4} {periode:12} {0:>6}  aucun signal")
            continue
        p, bm = b
        print(f"{nom:4} {periode:12} {st['n']:>6} {st['wr']:>8.1%} {st['moy_bps']:>8.1f}bp "
              f"{st['pf']:>5.2f} {st['total_pct']:>7.1f}% {st['max_dd_pct']:>6.1f}% | "
              f"{bm:>8.1f}bp {p:>10.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("symbol")
    ap.add_argument("--h4", action="store_true")
    ap.add_argument("--algos", default="A6,A5,B,C,D")
    ap.add_argument("--sims", type=int, default=100)
    a = ap.parse_args()
    SIMS = a.sims
    df = charger(a.csv, a.h4)
    gran = int(pd.Series(df.index).diff().median().total_seconds() // 60)
    print(f"\n=== {a.symbol} — {len(df)} bougies de {gran} min "
          f"({df.index[0].date()} → {df.index[-1].date()}), coût {COUTS[a.symbol] * 1e4:.1f} bp/trade ===")
    evaluer(df, a.symbol, a.algos.split(","), gran)
