# 🔬 RESEARCH — Index principal

> Recherche documentaire pour enrichir le bot Deriv (mission 4/5).
> **Sources uniquement** — pas de code modifié.

---

## 📊 Tableau de bord des sources

| Domaine | Fichier | Sources validées | Statut |
|---------|---------|------------------|--------|
| **Forex / Crypto** | `sources/forex_crypto/sources.md` | 16 | ✅ |
| **Synthétiques Deriv** | `sources/deriv_synthetic/sources.md` | 6 | ✅ |
| **Data (datasets/APIs)** | `sources/data/sources.md` | 10 | ✅ |
| **Papers & repos** | `sources/papers/sources.md` | 18 | ✅ |
| **Features inventory** | `notes/features_inventory.md` | — | ✅ |
| **Méthodes modèles** | `notes/model_methods.md` | — | ✅ |
| **Backtest** | `notes/backtest_method.md` | — | ✅ |
| **Risque & vote** | `notes/risk_and_voting.md` | — | ✅ |
| **Exclues** | `sources/excluded/sources.md` | — | ✅ |

---

## 🎯 Priorités intégration (P0 / P1 / P2)

### P0 — Immédiat (sans dette)

| Ajout | Source | Tags | Dette |
|-------|--------|------|-------|
| Reporting HTML post-trade | `quantstats` | `risk`, `monitor` | Nulle |
| Métriques live (Sharpe, drawdown) | `quantstats` | `monitor` | Nulle |
| Backtest walk-forward amélioré | `OmashelCap`, `MarketRegimeTrader` | `risk` | Faible |

### P1 — Après validation backtest

| Ajout | Source | Tags | Dette |
|-------|--------|------|-------|
| Features XGBoost complémentaires (CCI, Williams %R, SAR) | `Stock-Prediction-Models`, `ta` lib | `feature` | Faible |
| Hyperopt / Optuna | `freqtrade` (inspiration) | `model` | Moyenne |
| LSTM PyTorch (tsai) | `tsai` | `model` | Moyenne |

### P2 — Plus tard

| Ajout | Source | Tags | Dette |
|-------|--------|------|-------|
| Regime-aware risk (Bear → réduire stake) | `MarketRegimeTrader` | `risk` | Moyenne |
| Monte Carlo risk | `quantstats` | `risk` | Nulle |
| Sentiment / macro | FRED/ECB (features) | `feature` | Haute |

---

## 📁 Structure

```
research/
├── index.md                    ← ce fichier
├── sources/
│   ├── forex_crypto/           ✅ validées
│   ├── deriv_synthetic/        ✅ validées
│   ├── data/                   ✅ datasets + APIs
│   ├── papers/                 ✅ repos + papers
│   ├── repos/                  ✅ vue consolidée
│   └── excluded/               ✅ ce qu'on rejette
└── notes/
    ├── features_inventory.md   ✅ features actuelles + candidates
    ├── model_methods.md        ✅ 5 modèles + améliorations
    ├── backtest_method.md      ✅ pratiques + cibles
    └── risk_and_voting.md      ✅ risque + vote
```

---

## 🔗 Références maîtresses (à garder sous la main)

| Ressource | ★ | Usage |
|-----------|---|-------|
| `Machine Learning for Trading` (Stefan Jansen) | 20.9k | Pipeline complète |
| `awesome-quant` | ~18k | Toutes libs quant |
| `deriv-api` (officiel) | 76 | Doc API Deriv |
| `api.deriv.com` | — | Doc officielle WebSocket |

---

*Recherche documentaire uniquement — aucun code n'a été modifié par cette passe.*
*Mise à jour : 10/09/2026*
