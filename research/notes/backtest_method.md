# 📊 Méthode de backtesting

> Comment valider nos modèles sans data leakage ni surajustement.

---

## ✅ Pratiques actuelles

| Pratique | État | Backing |
|----------|------|---------|
| Split chronologique 80/20 (pas de shuffle) | ✅ | `MarketRegimeTrader`, `ML4T` |
| Early stopping sur validation set | ✅ | `freqtrade`, Keras |
| Walk-forward (backtest glissant) | ✅ (basique) | `OmashelCap`, `MarketRegimeTrader` |

---

## 🔜 Améliorations backingées

### Walk-forward avancé (WFA)
- **Source** : `kernc/backtesting.py` (MIT, ★7.5k)
- **Principe** : rolling windows, retrain périodique, évaluation out-of-sample continu
- **Intérêt** : valider que notre win rate 4/5 tient sur plusieurs périodes

### Purging & embargo
- **Source** : Marcos López de Prado (papers arXiv — non récupérés)
- **Principe** : supprimer les données trop proches du train pour éviter la contamination
- **Intérêt** : critique pour séries temporelles financières

### Combinatorial Purged Cross-Validation (CPCV)
- **Source** : `mlfinlab` (propriétaire)
- **Principe** : purging + combinaisons de folds
- **Intérêt** : backtest ultra-robuste (réserve pour plus tard)

### Monte Carlo
- **Source** : `ranaroussi/quantstats` (Apache-2.0)
- **Principe** : simulations de trajectoires pour estimer risque de ruine
- **Intérêt** : complément métriques classiques

---

## 🎯 Métriques cibles

| Métrique | Cible | Raison |
|----------|-------|--------|
| **Win rate** | > 55.6% | Breakeven options binaires (~85% payout) |
| **Profit Factor** | > 1.3 | Gains / pertes |
| **Max Drawdown** | < 15% | Risque acceptable |
| **Sharpe Ratio** | > 1.0 | Rendement/risque |
| **Expectancy** | > 0.1$ / trade | Rentabilité par trade |

---

## ⚠️ Pièges à éviter

1. **Look-ahead bias** : ne jamais utiliser des données futures dans les features
2. **Surajustement** : ne pas optimiser les hyperparams sur le test set
3. **Sur-trading** : le vote 4/5 est TRÈS sélectif — c'est volontaire (qualité > quantité)
4. **Transaction costs** : inclure spread/slippage dans le backtest
5. **Survivorship bias** : ne pas backtester que sur les paires qui "marchaient bien"
