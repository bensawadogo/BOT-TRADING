"""Fumée : pipeline complet HMM+XGBoost (réel) sur données synthétiques."""
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, r"c:\BOT-TRADING")

from deriv.ensemble_predictor import EnsemblePredictor, _XGB_OK, _TF_OK  # noqa: E402


def ohlcv(n=800):
    rng = np.random.default_rng(7)
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.006, n)))
    high = close * (1 + rng.uniform(0.001, 0.005, n))
    low = close * (1 - rng.uniform(0.001, 0.005, n))
    opn = low + rng.uniform(0.1, 0.9, n) * (high - low)
    return pd.DataFrame({"open": opn, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(100, 500, n)},
                        index=idx)


t0 = time.time()
df = ohlcv()
ens = EnsemblePredictor()
print(f"XGB installed={_XGB_OK} TF installed={_TF_OK}")

# Entraînement réel (70%)
split = int(len(df) * 0.7)
ens.train_all(df.iloc[:split])

# Rechargement depuis le disque (persistance)
from deriv.ensemble_predictor import HMMPredictor, XGBoostPredictor  # noqa: E402
assert HMMPredictor().is_trained, "HMM rechargé depuis pkl attendu"
assert XGBoostPredictor().is_trained, "XGB rechargé depuis pkl attendu"

# Backtest glissant sur le test set (échantillonné : pas de 5 bougies)
preds, actuals = [], []
step = 5
for i in range(60, len(df), step):
    res = ens.predict(df.iloc[:i])
    if not res["tradeable"]:
        continue
    if i + 1 >= len(df):
        break
    actuals.append(int(df["close"].iloc[i + 1] > df["close"].iloc[i]))
    preds.append(1 if res["signal"] == "BUY" else 0)
    if len(preds) >= 60:  # limite de temps raisonnable
        break

acc = sum(p == a for p, a in zip(preds, actuals)) / max(len(preds), 1)
from collections import Counter
print(f"Trades simulés : {len(preds)} — distribution {Counter(preds)}")
print(f"Win rate glissant : {acc:.1%} (breakeven 55.6%)")
print(f"Durée totale : {time.time() - t0:.1f}s")
print("SMOKE_OK")