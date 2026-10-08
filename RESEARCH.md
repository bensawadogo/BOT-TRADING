# 🔬 RESEARCH — Sources solides pour enrichir le bot Deriv (mission 4/5)

> Recherche par **Google Dorks / GitHub Search API** en 3 vagues de "sous-agents"
> (collecte brute → qualification README → vérif stars/licences). 10/09/2026.
> Cible : enrichir `c:\BOT-TRADING` (HMM → LSTM → XGBoost → Kalman → RSI, vote 4/5).

---

## 🏆 Top recommandations (à cloner en priorité)

| # | Projet | ★ | Licence | Pourquoi c'est la pépite |
|---|--------|---|---------|--------------------------|
| 1 | [`huseinzol05/Stock-Prediction-Models`](https://github.com/huseinzol05/Stock-Prediction-Models) | ~8 700 | MIT | ### D — XGBoost / ensembles
- **`huseinzol05/Stock-Prediction-Models`** — XGBoost/LGBM/Catboost sur features techniques : notre `FEATURE_COLS` (RSI, MACD, BB, ADX…) y trouve ses équivalents. ⭐ **10/10**
- **`anknpolley123/BTCDump`** — Ensemble XGBoost+RF+GBM en live avec refresh : bon exemple de pipeline "auto-prediction" temps réel. ⭐ **5/10** (exécution asset-picking)

### E — Backtesting / validation
- **`kernc/backtesting.py`** (★~7,5k, MIT, actif) — Idéal pour transformer notre backtest glissant en framework propre (métriques, walk-forward). ⭐ **9/10**
- **`polakowo/vectorbt`** (★9 054, licence à vérifier, actif) — Backtest **vectorisé ultra-rapide** pour scanner des milliers de paramètres de stake/durée sur du 60s. ⭐ **8/10**
- **`mementum/backtrader`** (★23 221, GPL-3.0, maintenance lente) — Framework historique riche, mais GPL et vétuste. ⭐ **7/10**
- **`freqtrade/freqtrade`** (★54 242, GPL-3.0, très actif) — Bot complet avec hyperopt (optimisation de paramètres), walk-forward, paper trading, Telegram. Référence d'ingénierie, mais crypto + GPL (pas de dérivé fermé). ⭐ **9/10** (inspiration, pas à copier)

### F — Risk management / position sizing
- **`OmashelCap/risk/`** — Fixed-fractional sizing (1-2%), cap de stake, **circuit breaker** (trips sur drawdown journalier, reset sur win), tests unitaires. Extrêmement proche de notre `RiskManager`. ⭐ **9/10**
- **`MarketRegimeTrader/risk/`** — VaR/CVaR, max drawdown, métriques par régime. ⭐ **8/10**
- **`0xemmkty/QuantMuse`** — "AI-powered risk management" en vrac, moins structuré. ⭐ **5/10**

---

## 🧠 Ce qu'on retire concrètement pour notre bot

| Briques manquantes / à enrichir | Source(s) recommandée(s) | Compatibilité stack |
|---|---|---|
| **Backtest walk-forward sérieux** (rolling windows) | `MarketRegimeTrader/backtesting` + `OmashelCap/backtest` | pandas / hmmlearn ✓ |
| **LSTM sans TensorFlow** (Py3.14) | `tsai` (PyTorch) ou `Stock-Prediction-Models` | PyTorch / fastai |
| **Circuit breaker + sizing robuste** | `OmashelCap/risk/` | pur Python ✓ |
| **Métriques de perf** (Sharpe, drawdown, VaR) | `MarketRegimeTrader`, `backtesting.py` | pandas ✓ |
| **Features XGBoost de référence** | `Stock-Prediction-Models` | `ta` ✓ |
| **Idées de stratégies binaires** | `ORSTAC/Bots_XML/` | XML DBot → lire, pas importer |
| **Design hyperopt / WFA** | `freqtrade` | inspiration seulement |

## ⚖️ Licences — attention
- **MIT** (copie compatible) : OmashelCap, MarketRegimeTrader, ORSTAC, backtesting.py, Stock-Prediction-Models, awesome-quant.
- **Apache-2.0** : tsai.
- **GPL-3.0** (contamination si intégration) : freqtrade, backtrader, KittenCN/stock_prediction.
- **À vérifier avant usage** : vectorbt ("Other" sur l'API), QuantMuse.

## 🧭 Prochaines étapes suggérées
1. Cloner `OmashelCap` et `MarketRegimeTrader` → lire `risk/`, `backtest/`, `tda/`.
2. Évaluer `tsai` pour le modèle 3 (LSTM PyTorch) en prévision de l'installation Python 3.12.
3. Utiliser `Stock-Prediction-Models` pour benchmarker notre `build_features` + nos win-rates.
4. Ne **jamais** intégrer de code GPL dans le bot sans avoir décidé de la licence finale du projet.

---
*Recherche documentaire uniquement — aucun code n'a été modifié par cette passe.*
| 2 | [`iamMashel/OmashelCap`](https://github.com/iamMashel/OmashelCap) | 5 | MIT | **Bot Deriv synthétiques production-grade** : modes mock + réel, structure de marché (confluence), backtest **walk-forward sur vraies bougies Deriv**, risque **fixed-fractional + circuit breaker**, alertes Telegram, tests pytest. Structure exemplaire (`risk/`, `backtest/`, `trading/`, `notify/`). |
| 3 | [`0x596173736972/MarketRegimeTrader`](https://github.com/0x596173736972/MarketRegimeTrader) | 18 | MIT | **Exactement notre brique HMM** : régimes Bull/Bear/Range, **walk-forward validation**, **TDA**, 30+ métriques (VaR/CVaR/drawdown), backtest avec coûts. Stack compatible (hmmlearn, optuna). |
| 4 | [`kernc/backtesting.py`](https://github.com/kernc/backtesting.py) | ~7 500 | MIT | Backtesting propre et léger, facile à adapter à des contrats à durée fixe (analyse par événement). |
| 5 | [`timeseriesAI/tsai`](https://github.com/timeseriesAI/tsai) | ~5 500 | Apache-2.0 | **Alternative PyTorch au LSTM TensorFlow** (indisponible sur Python 3.14 !). SOTA en deep learning temporel. |

## 📚 Références "liste maîtresse" (à garder sous la main)

| Projet | ★ | Licence | Note |
|--------|---|---------|------|
| [`wilsonfreitas/awesome-quant`](https://github.com/wilsonfreitas/awesome-quant) | ~18 000 | MIT | Toutes les libs quant Python existantes, triées par catégorie. |
| [`grananqvist/Awesome-Quant-Machine-Learning-Trading`](https://github.com/grananqvist/Awesome-Quant-Machine-Learning-Trading) | 4 019 | MIT | ML appliqué au trading : papers, implémentations, ressources. |
| [`deriv-com/deriv-api`](https://github.com/deriv-com/deriv-api) | 76 | MIT | API officielle WebSocket deriv.app (JS). Référence de vérité pour valider payloads/endpoints (doc + exemples). |

## 🎯 Qualification détaillée par thème

### A — Deriv / options binaires
- **`OmashelCap`** (★5, MIT, actif) — Le plus proche de notre cible. Idées fortes : *mock mode* pour tester sans risque, *walk-forward engine* sur candles Deriv, *circuit breaker* qui coupe après perte max, *Telegram signal-only*. ⭐ **9/10**
- **`alanvito1/ORSTAC`** (★204, MIT, actif, 177 forks) — 4 000+ bots XML pour DBot. C'est un **dump communautaire** (pas une lib Python) : à utiliser comme source d'idées de stratégies (martingale, moyennes mobiles, RSI). ⭐ **6/10**
- **`ezozu/MarketVision`** — Backtest d'arbitrage binaire stochastique + Martingale sur tick data. ⭐ **5/10** (risqué, à étudier avec recul)

### B — HMM / régimes & ensemble HMM+XGBoost
- **`MarketRegimeTrader`** (★18, MIT) — HMM + TDA + walk-forward + risk : la validation "out-of-sample" y est sérieuse (rolling windows, optimisation optuna). ⭐ **8/10**
- **`chilton-01/marl-trading-system`** (0★) — `MAPPO + GaussianHMM + XGBoost` : architecture multi-modèles proche de la nôtre (HMM régimes puis modèle tabulaire). ⭐ **7/10** (inspiration)
- **Notre choix actuel** (`hmmlearn.GaussianHMM` 3 états + `predict_proba`=Forward-Backward + `predict`=Viterbi) est validé par ces repos : mêmes patterns.

### C — LSTM / deep learning (remplace le TF manquant)
- **`tsai`** (★5,5k, Apache-2.0) — PyTorch/fastai SOTA time series. **LE candidat pour réimplémenter notre LSTM sans TensorFlow.** ⭐ **9/10**
- **`huseinzol05/Stock-Prediction-Models`** — réseaux récurrents (LSTM/GRU), attention, combos avec XGBoost, évaluations chiffrées. ⭐ **10/10**
- **`KittenCN/stock_prediction`** (★377, GPL-3.0) — LSTM + BERT/Transformers : utile pour du sentiment plus tard. ⭐ **6/10** (GPL → à éviter si code fermé)