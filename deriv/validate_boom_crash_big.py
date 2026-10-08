"""
════════════════════════════════════════════════════════════════════
TEST DRIFT BOOM/CRASH À GRANDE ÉCHELLE (10 000+ bougies)
deriv/validate_boom_crash_big.py
════════════════════════════════════════════════════════════════════

Objectif : décider si le drift post-spike Boom/Crash est un vrai edge.

Améliorations vs v1 :
  1. Historique 10 000 bougies (pagination epoch).
  2. Seuil de spike = QUANTILE 99.5% des moves (au lieu de 3×std fixe).
  3. Walk-forward MODÈLE BINAIRE (Rise/Fall) : win si le prix à
     l'expiration est du bon côté. Le seuil est calibré sur le SEGMENT
     D'ENTRAÎNEMENT uniquement (pas de fuite), puis appliqué au test.
  4. Statistiques : WR global, WR par fold, intervalle de Wilson,
     test binomial, stabilité (σ des folds), n trades total.

Usage : python deriv/validate_boom_crash_big.py --symbol CRASH500
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from math import sqrt

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BOOM_CRASH_BIG")


def _wilson(n: int, wins: int, z: float = 1.96) -> tuple:
    """Intervalle de Wilson 95% pour un win rate."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    marge = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - marge, centre + marge)


def _binomial_pvalue(n: int, wins: int, p0: float = 0.556) -> float:
    """P-value binomiale (H0 : WR <= 55.6%)."""
    from scipy.stats import binomtest
    res = binomtest(wins, n, p0, alternative="greater")
    return res.pvalue


def detect_spikes_idx(closes: np.ndarray, thr: float,
                      is_boom: bool) -> np.ndarray:
    """Indices i où la bougie i-1 → i est un spike (|move| >= thr)."""
    moves = np.diff(closes) / closes[:-1] * 100.0
    if is_boom:
        idx = np.nonzero(moves >= thr)[0] + 1
    else:
        idx = np.nonzero(moves <= -thr)[0] + 1
    return idx


def spike_threshold_quantile(closes: np.ndarray, is_boom: bool,
                             q: float = 0.995, min_thr: float = 0.08) -> float:
    """Seuil de spike = quantile q de |moves| (évite le bruit du 3×std)."""
    moves = np.abs(np.diff(closes) / closes[:-1] * 100.0)
    if moves.size < 50:
        return min_thr
    return max(min_thr, float(np.quantile(moves, q)))


def walk_forward_binaire(df: pd.DataFrame, symbol: str,
                         fold_size: int = 1500, horizon: int = 10,
                         min_trades_fold: int = 5,
                         quantile: float = 0.995) -> dict:
    """
    Walk-forward du modèle binaire.

    Chaque fold : on détecte les spikes dans le segment TEST avec un seuil
    calibré sur le segment TRAIN (quantile des moves du train),
    puis on mesure le WR de direction à l'expiration (horizon bougies).
    """
    closes = df["close"].values
    n = len(closes)
    is_boom = "BOOM" in symbol.upper()
    n_barrier = horizon

    results = []
    start = 100  # premier segment (warm-up)
    while start + fold_size + n_barrier < n:
        train_slice = closes[start: start + fold_size]
        test_start = start + fold_size
        test_end = min(test_start + fold_size, n - n_barrier)
        if test_end <= test_start:
            break

        # Seuil calibré sur le train uniquement
        thr = spike_threshold_quantile(train_slice, is_boom, q=quantile)

        # Détection des spikes dans le test
        spikes = detect_spikes_idx(closes[test_start: test_end + 1], thr, is_boom)
        n_sig = 0
        n_win = 0
        for i in spikes:
            abs_i = test_start + i
            if abs_i + n_barrier >= n:
                continue
            diff = closes[abs_i + n_barrier] - closes[abs_i]
            n_sig += 1
            if (is_boom and diff < 0) or (not is_boom and diff > 0):
                n_win += 1

        if n_sig >= min_trades_fold:
            results.append({
                "fold": len(results) + 1,
                "n_trades": n_sig,
                "win_rate": round(n_win / n_sig, 4),
                "thr": round(thr, 4),
            })
        start = test_start + 1

    if not results:
        return {"n_trades": 0, "win_rate": 0.0, "message": "Aucun fold valide"}

    tot_trades = sum(r["n_trades"] for r in results)
    tot_wins = sum(int(round(r["win_rate"] * r["n_trades"])) for r in results)
    wr = tot_wins / tot_trades
    lo, hi = _wilson(tot_trades, tot_wins)
    wrs = [r["win_rate"] for r in results]
    std = float(np.std(wrs))

    return {
        "n_trades": tot_trades,
        "n_wins": tot_wins,
        "win_rate": round(wr, 4),
        "wilson_lo": round(lo, 4),
        "wilson_hi": round(hi, 4),
        "p_value": round(float(_binomial_pvalue(tot_trades, tot_wins)), 4),
        "n_folds": len(results),
        "win_rate_std": round(std, 4),
        "wr_par_fold": results,
        "breakeven": 0.556,
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="CRASH500",
                   help="BOOM500, BOOM1000, CRASH500, CRASH1000...")
    p.add_argument("--count", type=int, default=10000)
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--horizon", type=int, default=10,
                   help="expiration (bougies) du contrat binaire")
    p.add_argument("--fold", type=int, default=1500,
                   help="taille de segment pour le WFO")
    p.add_argument("--quantile", type=float, default=0.995,
                   help="quantile des |moves| pour le seuil de spike "
                        "(0.995 = strict, 0.99 = plus de trades)")
    args = p.parse_args()

    from deriv.data_collector import DerivDataCollector

    print(f"\nChargement de {args.count} bougies {args.symbol} "
          f"(granularité {args.granularity}s, pagination)...")
    collector = DerivDataCollector(args.symbol, args.granularity)
    await collector.connect()
    df = await collector.get_candle_history_full(args.count)
    print(f"Données chargées : {len(df)} bougies\n")

    result = walk_forward_binaire(df, args.symbol,
                                  fold_size=args.fold, horizon=args.horizon,
                                  quantile=args.quantile)

    print("=" * 66)
    print(f"WALK-FORWARD BINAIRE — {args.symbol} (horizon {args.horizon})")
    print("=" * 66)
    if result.get("n_trades", 0) == 0:
        print(result.get("message", "Aucun trade"))
        return result

    print(f"Win Rate OOS      : {result['win_rate']:.1%} "
          f"sur {result['n_trades']} trades / {result['n_folds']} folds")
    print(f"Intervalle Wilson : [{result['wilson_lo']:.1%}, "
          f"{result['wilson_hi']:.1%}] (95%)")
    print(f"p-value (H0:≤55.6%) : {result['p_value']:.4f}")
    print(f"Stabilité (σ folds)  : {result['win_rate_std']:.1%}")
    print(f"Breakeven Rise/Fall  : {result['breakeven']:.1%}")
    print("\nDétail par fold (seuil calibré sur train) :")
    for f in result["wr_par_fold"]:
        bar = "🟢" if f["win_rate"] >= result["breakeven"] else "🔴"
        print(f"  Fold {f['fold']:2d}: {bar} {f['win_rate']:.1%} "
              f"({f['n_trades']} trades, seuil {f['thr']:.2f}%)")
    print("=" * 66)

    wr = result["win_rate"]
    sig = result["p_value"] < 0.05
    stable = result["win_rate_std"] < 0.10
    enough = result["n_trades"] >= 100
    verdict = (
        "✅ EDGE RÉEL CONFIRMÉ → candidat démo réelle 1$"
        if (wr >= 0.556 and sig and stable and enough)
        else "❌ PAS d'edge significatif → rester en mode historique"
    )
    print(f"\n{verdict}")
    print("  (critères : WR≥55.6% ET p<0.05 ET σ<10% ET ≥100 trades)")
    return result


if __name__ == "__main__":
    asyncio.run(main())