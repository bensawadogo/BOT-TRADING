"""Preuve du biais de base rate — CRASH500 M1.

Hypothese : le WR 89.67% de XGBoost/HMM n'est PAS une performance mais
le simple taux de bougies haussieres (baseline "toujours UP") sur les
memees fenetres de test que le WFO.

Si up_rate(test windows) == 0.8967 -> les modeles predisent toujours UP
et le WIN RATE mesure la derive de l'indice, pas une competence.
"""
import sys, asyncio
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from deriv.data_collector import DerivDataCollector
from deriv.walk_forward_optimizer import WalkForwardOptimizer


async def main():
    c = DerivDataCollector('CRASH500', 60)
    await c.connect()
    df = await c.get_candle_history_full(target_count=5000)
    await c.close()

    # Target EXACTE utilisee par _evaluate : close[i] > close[i-1]
    actual = (df['close'].diff() > 0).astype(int).iloc[1:]
    print(f"bougies            : {len(df)}")
    print(f"up_rate GLOBAL     : {actual.mean():.4f}   (down_rate {1-actual.mean():.4f})")

    # Memes folds que le WFO (train 400 + embargo 30 + test 100, pas 100)
    plan = WalkForwardOptimizer.folds_plan(len(df), 400, 100, 30, 100, 1)
    print(f"n_folds            : {len(plan)}")

    ups, per_fold = [], []
    for fold in plan:
        seg = df.iloc[fold['test_start']:fold['test_end']]
        seg_actual = (seg['close'].diff() > 0).astype(int).iloc[1:]
        per_fold.append(seg_actual.mean())
        ups.extend(seg_actual.tolist())

    pooled = float(np.mean(ups))
    rd = np.round(per_fold, 4)
    print(f"preds collectees   : {len(ups)}")
    print(f"up_rate POOLE test : {pooled:.4f}   (45 x 99 = {45*99})")
    print(f"min/max folds      : {min(rd):.4f} / {max(rd):.4f}")
    print(f"ecart-type folds   : {np.std(per_fold):.4f}")

    print("\nWR observe WFO (JSON) : XGBoost 0.8967 / HMM 0.8967")
    print(f"baseline toujours UP  : {pooled:.4f}")
    if abs(pooled - 0.8967) < 0.005:
        print("VERDICT : BIAIS CONFIRME — le 'win rate' est la derive de l'indice,")
        print("          pas une competence predictive (modeles = toujours UP).")
    else:
        print("VERDICT : ecart notable — a investiguer autrement.")


asyncio.run(main())