"""
Test 4/2 — tendances très claires (faible bruit) + vérification labellisation.
"""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import numpy as np, pandas as pd
from deriv.ensemble_predictor import EnsemblePredictor, build_features

def make_trend(n, start, direction, noise=0.001):
    """direction: +1 haussier, -1 baissier, 0 range"""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, n)
    if direction > 0:
        close = start * (1 + 0.15 * t)  # +15% linéaire
    elif direction < 0:
        close = start * (1 - 0.15 * t)  # -15% linéaire
    else:
        close = start * (1 + 0.02 * np.sin(2*np.pi*t*3))  # range oscillant
    close += rng.normal(0, noise*start, n)  # bruit additif faible
    high = close * (1 + abs(rng.normal(0, 0.001, n)))
    low = close * (1 - abs(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    volume = rng.integers(100, 200, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"open":open_,"high":high,"low":low,"close":close,"volume":volume}, index=idx)

scenarios = {
    "BUY":  make_trend(600, 1.0800, +1, noise=0.0005),
    "SELL": make_trend(600, 1.0800, -1, noise=0.0005),
    "RANGE":make_trend(600, 1.0800,  0, noise=0.0005),
}

for name, df in scenarios.items():
    print(f"\n=== {name} ===")
    ens = EnsemblePredictor()
    ens.train_all(df)
    res = ens.predict(df)
    d = res["detail"]
    votes = [d[k]["vote"] for k in d]
    buy = sum(1 for v in votes if v >= 0.7)
    sell = sum(1 for v in votes if v <= 0.3)
    hold = sum(1 for v in votes if 0.3 < v < 0.7)
    print(f"  Signal={res['signal']} BUY={buy} SELL={sell} HOLD={hold}")
    print(f"  HMM={d['HMM']['vote']:.1f} XGB={d['XGBoost']['vote']:.1f} LSTM={d['LSTM']['vote']:.1f} "
          f"Kalman={d['Kalman']['vote']:.1f} RSI={d['RSI']['vote']:.1f} Trend={d['TrendStrength']['vote']:.1f}")

