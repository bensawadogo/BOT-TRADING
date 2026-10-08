# 📋 Inventaire des features

> Ce qu'on a déjà dans `build_features` et ce qu'on peut ajouter (avec backing).

---

## ✅ Features actuelles (`deriv/ensemble_predictor.py`)

| Feature | Type | Backing |
|---------|------|---------|
| `rsi_14`, `rsi_7` | Momentum | Standard Wilder, backed by `ta` lib + littérature |
| `roc_10` | Momentum | Standard, backed by `ta` lib |
| `mfi_14` | Volume/Momentum | Standard Chaikin MFI |
| `ema_9`, `ema_21`, `ema_50` | Tendance | Standard |
| `macd_diff` | Tendance | Standard |
| `adx_14` | Tendance | Standard |
| `bb_pct`, `bb_width` | Volatilité | Standard Bollinger |
| `atr_14` | Volatilité | Standard Wilder |
| `log_ret`, `log_ret_5` | Retour | Standard |
| `volatility` | Volatilité | Std dev des log returns |
| `kalman`, `kalman_delta` | Tendance | Filtre Kalman simple |
| `ema_ratio`, `price_ema50` | Tendance | Ratios dérivés |

**Total : 15 features** — déjà un feature set robuste et backingé.

---

## 🔜 Features candidates (avec backing)

| Feature | Source | Tags | Complexité | Intérêt |
|---------|--------|------|------------|---------|
| **OBV / Volume Ratio** | `Stock-Prediction-Models` | `feature` | Basse | Confirmer momentum avec volume |
| **CCI (Commodity Channel Index)** | `ta` lib | `feature` | Basse | Détection surachat/surcharge |
| **Williams %R** | `ta` lib | `feature` | Basse | Complément RSI |
| **Parabolic SAR** | `ta` lib | `feature` | Basse | Stop & reversal |
| **Ichimoku (Conversion/Base)** | `ta` lib | `feature` | Moyenne | Tendance multi-frame |
| **Keltner Channels** | `ta` lib | `feature` | Basse | Volatilité alternative BB |
| **Donchian Channels** | `ta` lib | `feature` | Basse | Breakout detection |
| **Pivot Points (S1/R1)** | `Stock-Prediction-Models` | `feature` | Basse | Support/résistance |
| **Order Book Imbalance** | `freqtrade` (inspiration) | `feature` | Haute | Microstructure (non dispo sur synthétiques) |

---

## ❌ Features rejetées (surajustement / pas de backing)

| Feature | Raison |
|---------|--------|
| Astrologie / planètes | Aucun backing |
| "Sentiment Twitter" sans dataset | Pas de dataset public fiable |
| Indicateurs custom sans formule | Pas de backing |
