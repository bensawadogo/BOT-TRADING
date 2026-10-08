"""Diagnostic XGBoost : pourquoi vote-t-il toujours 0.5 ?"""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import numpy as np, pandas as pd
from deriv.ensemble_predictor import build_features, FEATURE_COLS, XGBoostPredictor

np.random.seed(42)
n = 500

# Scenario avec tendance marquee
ret = np.linspace(0.0003, 0.0008, n) + np.random.normal(0, 0.001, n)
close = 1.1000 * np.exp(np.cumsum(ret))
high = close * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low  = close * (1 - np.abs(np.random.normal(0, 0.0005, n)))
open_ = close + np.random.normal(0, 0.0003, n)
volume = np.random.randint(10, 100, n)

df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
feat_df = build_features(df)

print("Features shape:", feat_df[FEATURE_COLS].shape)
print("Target distribution:", feat_df["target"].value_counts().to_dict())

# Test avec le modele actuel
xgb = XGBoostPredictor()
if xgb.model is not None:
    X = feat_df[FEATURE_COLS].iloc[[-1]].values
    X_scaled = xgb.scaler.transform(X)
    proba = float(xgb.model.predict_proba(X_scaled)[0][1])
    print("Proba XGBoost (modele actuel):", round(proba, 3))
    print("Vote:", 1 if proba > 0.58 else 0 if proba < 0.42 else 0.5)
else:
    print("Pas de modele XGBoost charge")

# Test: probas sur tout le dataset
if xgb.model is not None:
    X_all = feat_df[FEATURE_COLS].values
    X_all_scaled = xgb.scaler.transform(X_all)
    probas = xgb.model.predict_proba(X_all_scaled)[:, 1]
    print("\nProbas sur tout le dataset:")
    print("  min:", round(probas.min(), 3))
    print("  max:", round(probas.max(), 3))
    print("  mean:", round(probas.mean(), 3))
    print("  std:", round(probas.std(), 3))
    print("  > 0.58:", (probas > 0.58).sum(), "/", len(probas))
    print("  < 0.42:", (probas < 0.42).sum(), "/", len(probas))
    print("  0.42-0.58 (neutre):", ((probas >= 0.42) & (probas <= 0.58)).sum(), "/", len(probas))
