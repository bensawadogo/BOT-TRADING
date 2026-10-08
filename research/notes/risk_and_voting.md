# 🛡️ Risque & Vote

> Logique de gestion du risque et de vote majoritaire.

---

## ✅ Logique actuelle

### Vote majoritaire qualifié (4/5)
- **Règle** : un trade ne se place QUE si ≥4 modèles sur 5 sont d'accord
- **Modèles** : HMM, XGBoost, LSTM, Kalman, RSI Divergence
- **Votes** : 0 (SELL), 1 (BUY), 0.5 (HOLD/neutre)
- **Verrou** : `MIN_VOTES_TO_TRADE = 4` (constant, testé, non modifiable sans revue)

### Risk Manager (`deriv/risk_manager.py`)
- **Stake max** : `AUTO_STAKE` (défaut 2$)
- **Max risk pct** : `MAX_RISK_PCT` (défaut 2% du capital)
- **Daily stop** : `DAILY_STOP_PCT` (défaut 5%)
- **Max consecutive losses** : `MAX_CONSECUTIVE` (défaut 3)
- **Cooldown** : après pertes consécutives
- **Mode démo** : `DERIV_ACCOUNT_TYPE=demo` (verrouillé)

---

## 🔜 Améliorations backingées

### Position sizing avancé
- **Source** : `OmashelCap` (MIT), `mlfinlab` (bet sizing)
- **Méthodes** :
  - Fixed-fractional (1-2% du capital)
  - Kelly criterion (fraction optimale)
  - Volatility-targeting (ajuster selon ATR)
- **Intérêt** : optimiser le rendement/risque

### Circuit breaker intelligent
- **Source** : `OmashelCap` (trips sur drawdown journalier, reset sur win)
- **Principe** : couper automatiquement après X pertes ou Y% drawdown
- **Intérêt** : protection capital

### VaR / CVaR
- **Source** : `MarketRegimeTrader` (30+ métriques)
- **Principe** : estimer la perte maximale à un seuil de confiance
- **Intérêt** : risk monitoring avancé

### Regime-aware risk
- **Source** : `MarketRegimeTrader` (TDA + régimes)
- **Principe** : ajuster le risque selon le régime (Bear → réduire, Bull → augmenter)
- **Intérêt** : adapter la taille de position au contexte

---

## 🎯 Règles d'or (ne pas toucher)

| Règle | Raison |
|-------|--------|
| MIN_VOTES_TO_TRADE = 4 | Qualité > quantité. Baisser à 3 augmente les faux signaux |
| DERIV_ACCOUNT_TYPE = demo par défaut | Protection argent réel |
| AUTO_STAKE ≤ 2% capital | Protection drawdown |
| DAILY_STOP_PCT = 5% | Protection journalière |

---

## ⚠️ Pièges à éviter

1. **Baisser le seuil de vote** pour "plus de trades" → faux signaux → pertes
2. **Augmenter le stake** après pertes (martingale) → ruine certaine
3. **Ignorer le daily stop** → drawdown incontrôlé
4. **Passer en réel avant validation** → risque financier réel
