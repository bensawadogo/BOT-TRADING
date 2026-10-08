# 🧠 Méthodes de modélisation

> Référence pour nos 5 modèles et ce qu'on peut améliorer.

---

## 1️⃣ HMM (Baum-Welch + Forward-Backward + Viterbi)

**État actuel** : `HMMPredictor` avec `GaussianHMM(n_components=3)`, labels auto.

**Backing** :
- `MarketRegimeTrader` (★18, MIT) — même architecture
- Papers arXiv "regime switching finance" (non récupérés — voir notes)
- `hmmlearn` documentation officielle

**Améliorations possibles** :
- Régimes adaptatifs (BIC pour choisir n_components)
- HMM avec covariances tied vs full
- Features HMM : ajouter le volume proxy + ATR

---

## 2️⃣ XGBoost (features tabulaires)

**État actuel** : 15 features, early stopping, split 80/20 chronologique.

**Backing** :
- `Stock-Prediction-Models` — XGBoost/LGBM/Catboost sur features techniques (⭐10/10)
- `ML4T` — gradient boosting + features engineering
- `mlfinlab` — feature importance + bet sizing

**Améliorations possibles** :
- Hyperopt / Optuna pour hyperparamètres (inspiration `freqtrade`)
- Feature importance SHAP
- Sélection features récursive (RFE)

---

## 3️⃣ LSTM (deep learning temporel)

**État actuel** : Bidirectional LSTM, 30-window, EarlyStopping. **TensorFlow indisponible sur Python 3.14.**

**Backing** :
- `tsai` (Apache-2.0) — alternative PyTorch SOTA
- `Stock-Prediction-Models` — LSTM/GRU/Attention
- `deepdow` — deep learning + portfolio allocation

**Problème** : TF incompatible Python 3.14. **Solution** : migrer vers `tsai` (PyTorch) quand on sera en Python 3.12-3.13.

---

## 4️⃣ Kalman Filter (momentum)

**État actuel** : filtre simple (alpha-beta), trend + momentum.

**Backing** :
- `MarketRegimeTrader` — Kalman + régimes
- `freqtrade` — Kalman dans les indicateurs custom

**Améliorations possibles** :
- Kalman multidimensionnel (prix + volatilité)
- Extended Kalman Filter (EKF)

---

## 5️⃣ RSI Divergence

**État actuel** : détection divergence haussière/baissière sur 10 bougies.

**Backing** :
- Méthode classique ( Wilder + Murphy )
- `Stock-Prediction-Models` — indicateurs custom

**Améliorations possibles** :
- Divergence sur MACD
- Divergence multi-timeframe

---

## 🔀 Vote ensemble

**État actuel** : vote majoritaire qualifié (4/5 minimum). Modèles : vote 0 (SELL), 1 (BUY), 0.5 (HOLD).

**Backing** :
- Littérature "ensemble methods trading" (papers non récupérés)
- `chilton-01/marl-trading-system` — MAPPO + HMM + XGBoost (inspiration)

**Règle d'or** : MIN_VOTES_TO_TRADE = 4 (verrouillé, testé).
