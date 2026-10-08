"""Diagnostic: le 4/5 est-il atteignable avec les modèles entraînés ?"""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import numpy as np, pandas as pd
from deriv.ensemble_predictor import EnsemblePredictor

ens = EnsemblePredictor()
print("HMM trained:", ens.hmm.is_trained)
print("XGB trained:", ens.xgb.is_trained)
print("LSTM trained:", ens.lstm.is_trained)

def make_df(n, drift, seed=42):
    np.random.seed(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.05 + drift)
    high = close + np.abs(np.random.randn(n) * 0.05)
    low = close - np.abs(np.random.randn(n) * 0.05)
    open_ = close + np.random.randn(n) * 0.02
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1}, index=idx)

scenarios = [
    ("FORTE HAUSSE (drift +0.08)", make_df(80, 0.08, 1)),
    ("FORTE BAISSE (drift -0.08)", make_df(80, -0.08, 2)),
    ("MODEREE HAUSSE (drift +0.03)", make_df(80, 0.03, 3)),
    ("MODEREE BAISSE (drift -0.03)", make_df(80, -0.03, 4)),
    ("RANGE (drift 0.0)", make_df(80, 0.0, 5)),
    ("HAUSSE + HAUSSE alterne", make_df(80, 0.0, 6)),
]

tradeable_count = 0
for name, df in scenarios:
    res = ens.predict(df)
    t = "TRADE" if res["tradeable"] else "HOLD"
    if res["tradeable"]:
        tradeable_count += 1
    print(f"\n=== {name} ===")
    print(f"  Signal: {res['signal']} | {t} | buy={res['buy_votes']} sell={res['sell_votes']} hold={res['hold_votes']}")
    for m, d in res["detail"].items():
        print(f"    {m}: vote={d['vote']}")

print(f"\n=== RÉSUMÉ: {tradeable_count}/{len(scenarios)} scénarios tradeable ===")
if tradeable_count == 0:
    print("PROBLEME: jamais 4/5 — vérifier les seuils de vote des modèles")