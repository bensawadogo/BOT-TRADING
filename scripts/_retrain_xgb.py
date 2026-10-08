"""Reentrainer XGBoost avec equilibrage des classes."""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import numpy as np, pandas as pd
from deriv.ensemble_predictor import build_features, FEATURE_COLS, XGBoostPredictor
from sklearn.metrics import accuracy_score, classification_report

np.random.seed(42)
n = 800

# Dataset equilibre : moitie tendance haussiere, moitie baissiere
ret1 = np.linspace(0.0003, 0.0006, n//2) + np.random.normal(0, 0.0008, n//2)
ret2 = np.linspace(-0.0003, -0.0006, n//2) + np.random.normal(0, 0.0008, n//2)
ret = np.concatenate([ret1, ret2])

close = 1.1000 * np.exp(np.cumsum(ret))
high = close * (1 + np.abs(np.random.normal(0, 0.0005, n)))
low  = close * (1 - np.abs(np.random.normal(0, 0.0005, n)))
open_ = close + np.random.normal(0, 0.0003, n)
volume = np.random.randint(10, 100, n)

df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})
feat_df = build_features(df)

print("Target distribution:", feat_df["target"].value_counts().to_dict())

X = feat_df[FEATURE_COLS].values
y = feat_df["target"].values

# Split 80/20
split = int(len(X) * 0.8)
X_tr, X_val = X[:split], X[split:]
y_tr, y_val = y[:split], y[split:]

# Calculer scale_pos_weight
n_down = (y_tr == 0).sum()
n_up = (y_tr == 1).sum()
spw = n_down / n_up if n_up > 0 else 1.0
print("scale_pos_weight:", round(spw, 3))

# Entrainer avec equilibrage
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)

model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss",
    early_stopping_rounds=20,
)
model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

# Evaluer
y_pred = model.predict(X_val_s)
acc = accuracy_score(y_val, y_pred)
print("Accuracy validation:", round(acc, 3))
print(classification_report(y_val, y_pred, target_names=["DOWN", "UP"]))

# Test de calibration
probas = model.predict_proba(X_val_s)[:, 1]
print("\nCalibration:")
print("  min:", round(probas.min(), 3))
print("  max:", round(probas.max(), 3))
print("  mean:", round(probas.mean(), 3))
print("  > 0.58:", (probas > 0.58).sum(), "/", len(probas))
print("  < 0.42:", (probas < 0.42).sum(), "/", len(probas))
print("  neutre:", ((probas >= 0.42) & (probas <= 0.58)).sum(), "/", len(probas))

# Sauvegarder
import pickle, os
os.makedirs("deriv/models", exist_ok=True)
with open("deriv/models/xgb_deriv.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)
print("\nModele sauvegarde dans deriv/models/xgb_deriv.pkl")
