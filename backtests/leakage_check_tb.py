"""Check anti-fuite complet — target triple-barrière (P0-v2).

1. Détecteur de feature leakage (López de Prado) sur données RÉELLES :
   corr(feature[t], target[t]) vs corr(feature[t-1], target[t]).
2. Preuve runtime : les barrières d'un label TRAIN ne chevauchent JAMAIS
   le fold de test (build_labels est appelé sur le slice train seul →
   les labels à moins de `horizon` bougies de la fin sont éliminés).
3. Deux symboles : frxEURUSD H4 (sans drift) et CRASH500 M1 (audit).
"""
import asyncio

import numpy as np
import pandas as pd

from deriv.data_collector import DerivDataCollector
from deriv.ensemble_predictor import (
    FEATURE_COLS,
    build_features,
    build_labels,
)
from deriv.leakage_detector import detect_feature_leakage
from deriv.walk_forward_optimizer import WalkForwardOptimizer


def check_wfo_purge_barriers():
    """Aucune bougie-barrière d'un label TRAIN n'atteint le fold TEST."""
    rng = np.random.default_rng(7)
    n = 1200
    base = 1.1000 + np.cumsum(rng.standard_normal(n) * 0.001)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + np.abs(rng.standard_normal(n)) * 0.0005,
            "low": base - np.abs(rng.standard_normal(n)) * 0.0005,
            "close": base,
            "volume": np.ones(n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="4h"),
    )
    wfo = WalkForwardOptimizer()
    marge_min = 10**9
    n_folds = 0
    for fold in wfo.folds_plan(n):
        t0, t1 = fold["train"]
        ts = fold["test_start"]
        sub = df.iloc[t0:t1]
        labels = build_labels(
            sub, mode="triple_barrier", touch_mult=1.5, horizon=3,
        )
        if labels is None or labels.empty:
            continue
        n_folds += 1
        dernier = t0 + sub.index.get_loc(labels.index[-1])
        portee = dernier + 3          # dernière bougie-barrière (horizon=3)
        assert portee <= t1 - 1, (
            f"FUITE : barrières train (fold train=({t0},{t1}), "
            f"test_start={ts}) atteignent la bougie {portee} >= t1-1"
        )
        assert t1 <= ts, f"Chevauchement train/test : t1={t1} > test_start={ts}"
        marge_min = min(marge_min, ts - 1 - portee)
    print(f"\n[2] Purge/barrières WFO : {n_folds} folds vérifiés — "
          f"AUCUNE barrière train ne touche le test "
          f"(marge min = {marge_min} bougies)")


async def check_symbole(symbol: str, gran: int, count: int) -> int:
    c = DerivDataCollector(symbol, gran)
    await c.connect()
    df = await c.get_candle_history_full(target_count=count)
    await c.close()

    feat = build_features(df)
    labels = build_labels(
        df, mode="triple_barrier", touch_mult=1.5, horizon=3,
    )
    ft = feat.assign(target=labels.reindex(feat.index)).dropna(
        subset=["target"]
    )
    print(f"\n=== {symbol} {gran}s : {len(df)} bougies, "
          f"{len(ft)} lignes targetées (TB) ===")
    res = detect_feature_leakage(ft, FEATURE_COLS)
    print("Verdict :", res["verdict"])
    for f, d in res["detail"].items():
        if d.get("skipped"):
            print(f"  {f:15s} — non testable (constant/faible)")
        else:
            print(f"  {f:15s} corr(t0)={d['corr_t0']:+.4f} "
                  f"corr(lag)={d['corr_lag1']:+.4f} "
                  f"ratio={d['ratio']}x p={d['pvalue_t0']}")
    return res["n_leaked"]


async def main():
    fuites = 0
    fuites += await check_symbole("frxEURUSD", 14400, 1000)
    fuites += await check_symbole("CRASH500", 60, 1000)
    check_wfo_purge_barriers()
    print(f"\n=== TOTAL features fuitées : {fuites} ===")
    if fuites:
        raise SystemExit(1)


asyncio.run(main())
