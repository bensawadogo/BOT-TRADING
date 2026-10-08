"""Test reel des 4 modeles actifs."""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import numpy as np, pandas as pd
from deriv.ensemble_predictor import EnsemblePredictor, build_features

np.random.seed(42)
n = 300
ret = np.random.normal(0.0001, 0.001, n)
close = 1.1000 * np.exp(np.cumsum(ret))
high = close * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low  = close * (1 - np.abs(np.random.normal(0, 0.0005, n)))
open_ = close + np.random.normal(0, 0.0003, n)
volume = np.random.randint(10, 100, n)

df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})

ep = EnsemblePredictor()
print("Modeles: HMM=%s XGB=%s LSTM=%s" % (
    ep.hmm.model is not None,
    ep.xgb.model is not None,
    ep.lstm.model is not None,
))

result = ep.predict(df)
print("\n=== Prediction reelle ===")
print("Signal:", result["signal"])
print("BUY:", result["buy_votes"], "SELL:", result["sell_votes"], "HOLD:", result["hold_votes"])
print("Confiance:", result["confiance"])
print("Tradeable:", result["tradeable"])
print()
for name, d in result["detail"].items():
    print("  %s -> %s" % (name, d))

# Test 2: forcer un signal tres baissier
print("\n=== Test baissier ===")
ret2 = np.random.normal(-0.0005, 0.002, n)
close2 = 1.1000 * np.exp(np.cumsum(ret2))
high2 = close2 * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low2  = close2 * (1 - np.abs(np.random.normal(0, 0.0005, n)))
open2_ = close2 + np.random.normal(0, 0.0003, n)
df2 = pd.DataFrame({"open": open2_, "high": high2, "low": low2, "close": close2, "volume": volume})

result2 = ep.predict(df2)
print("Signal:", result2["signal"])
print("BUY:", result2["buy_votes"], "SELL:", result2["sell_votes"], "HOLD:", result2["hold_votes"])
print("Confiance:", result2["confiance"])
print("Tradeable:", result2["tradeable"])
for name, d in result2["detail"].items():
    print("  %s -> %s" % (name, d))

# Test 3: tendance haussiere marquee
print("\n=== Test haussier ===")
ret3 = np.linspace(0.0005, 0.001, n) + np.random.normal(0, 0.0008, n)
close3 = 1.1000 * np.exp(np.cumsum(ret3))
high3 = close3 * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low3  = close3 * (1 - np.abs(np.random.normal(0, 0.0005, n)))
open3_ = close3 + np.random.normal(0, 0.0003, n)
df3 = pd.DataFrame({"open": open3_, "high": high3, "low": low3, "close": close3, "volume": volume})

result3 = ep.predict(df3)
print("Signal:", result3["signal"])
print("BUY:", result3["buy_votes"], "SELL:", result3["sell_votes"], "HOLD:", result3["hold_votes"])
print("Confiance:", result3["confiance"])
print("Tradeable:", result3["tradeable"])
for name, d in result3["detail"].items():
    print("  %s -> %s" % (name, d))

print("\n=== Recapitulatif ===")
signals = [result["signal"], result2["signal"], result3["signal"]]
print("Signaux sur 3 scenarios:", signals)
print("Au moins 1 BUY ou SELL:", any(s in ("BUY", "SELL") for s in signals))
print("Regle 4/5 respectee: aucun trade sans 4 votes -> ",
      all((r["tradeable"] == False) or (r["buy_votes"] >= 4) or (r["sell_votes"] >= 4)
          for r in [result, result2, result3]))
