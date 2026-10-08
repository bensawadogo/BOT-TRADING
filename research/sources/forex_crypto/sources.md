# 📊 Forex / Crypto — Sources validées

> Sources avec backing (paper, repo public, dataset, doc API) pour les marchés réels.
> Tags : `data` · `feature` · `model` · `vote` · `risk` · `monitor` · `hold`

---

## 🟢 Validées avec backing

### Repos GitHub

| # | Nom | ★ | Licence | Tags | Apport | Intégration | Lien |
|---|-----|---|---------|------|--------|-------------|------|
| 1 | **huseinzol05/Stock-Prediction-Models** | ~8 700 | MIT | `feature`, `model` | XGBoost/LGBM/Catboost sur features techniques (RSI, MACD, BB, ADX…). Benchmark de référence. | Comparer notre `build_features` + win-rate | https://github.com/huseinzol05/Stock-Prediction-Models |
| 2 | **stefan-jansen/machine-learning-for-trading** | ~20 900 | MIT | `feature`, `model`, `risk`, `data` | Livre "Machine Learning for Trading" 3rd ed — donnéesAlgoSeek, features, gradient boosting, DL time series, risk management. Pipeline complet. | Référence majeure pour features XGBoost + backtest + risque | https://github.com/stefan-jansen/machine-learning-for-trading |
| 3 | **hudson-and-thames/mlfinlab** | ~4 900 | Propriétaire (community free) | `feature`, `model`, `risk` | Outils ML financiers : data structures, labeling, sampling, feature engineering, clustering, hyperparameter tuning, bet sizing. | Inspiration features + méthodes (pas copie code propriétaire) | https://github.com/hudson-and-thames/mlfinlab |
| 4 | **ranaroussi/quantstats** | ~7 600 | Apache-2.0 | `risk`, `monitor` | Analytics portfolio : Sharpe, drawdown, VaR, win rate, reporting HTML, Monte Carlo. | Reporting post-trade + métriques live | https://github.com/ranaroussi/quantstats |
| 5 | **freqtrade/freqtrade** | ~54 200 | GPL-3.0 | `monitor`, `risk`, `feature` | Bot crypto complet : hyperopt, walk-forward, paper trading, Telegram. Référence d'ingénierie. | Inspiration structure + monitoring (⚠️ GPL, pas copie) | https://github.com/freqtrade/freqtrade |
| 6 | **quantconnect/Lean** | ~21 600 | Apache-2.0 | `data`, `feature`, `risk` | Moteur quant multi-assets : pipeline data, indicateurs, gestion risque, optimisation. | Pattern pipeline data + indicateurs | https://github.com/quantconnect/Lean |
| 7 | **jankrepl/deepdow** | ~1 200 | Apache-2.0 | `model` | Deep learning + optimisation portfolio (PyTorch, cvxpylayers, Markowitz). | Inspiration allocation LSTM | https://github.com/jankrepl/deepdow |
| 8 | **borisbanushev/stockpredictionai** | ~5 600 | MIT | `model`, `feature` | GAN + LSTM + CNN, WGAN, features techniques. Notebook éducatif. | Référence LSTM architectures (qualitatif) | https://github.com/borisbanushev/stockpredictionai |

### APIs publiques (free / freemium)

| # | Nom | Tags | Limites / licence | Apport | Lien |
|---|-----|------|-------------------|--------|------|
| 9 | **Alpha Vantage** | `data` | 25 req/jour (free), clé requise. Y Combinator / NASDAQ backed. | Forex/crypto quotidien + intraday (1min). Remplacement possible de l'historique Deriv. | https://www.alphavantage.co/ |
| 10 | **Polygon.io** | `data` | Free tier limité, payant au-delà. Backé par des VCs. | Tick-level forex/crypto, aggregates (bougies), WebSocket. Backup data feed haute qualité. | https://polygon.io/ |
| 11 | **Twelve Data** | `data` | 800 req/jour (free), 700M req/jour en prod. | Time series forex/crypto, WebSocket, time zones. Alternative crédible. | https://twelvedata.com/ |
| 12 | **Binance Public API** | `data` | Aucune clé pour les endpoints publics (klines/aggTrades). Rate limit 1200 req/min. | OHLCV crypto spot/deriv 1s-1M, tick data via WebSocket. | https://binance-docs.github.io/apidocs/ |

### Datasets publics

| # | Nom | Tags | Format | Source | Lien |
|---|-----|------|--------|--------|------|
| 13 | **AlgoSeek SP500 Daily Bars** | `data` | Parquet | Données académiques (utilisées dans ML4T) | Via `stefan-jansen/machine-learning-for-trading` (data/README.md) |
| 14 | **Dukascopy Historical** | `data` | CSV | Tick data forex/crypto gratuit (banque suisse). | https://www.dukascopy.com/swiss/english/marketwatch/historical/ |
| 15 | **FRED (Federal Reserve Economic Data)** | `data` | CSV/JSON | Données macro US (taux, inflation, PIB) — features contextuelles. | https://fred.stlouisfed.org/ |
| 16 | **ECB Statistical Data Warehouse** | `data` | CSV/JSON | Taux EUR, Forex quotidien officiel (BCE). | https://sdw.ecb.europa.eu/ |

---

## 🔴 Rejetées (pas de backing)

| Nom | Raison |
|-----|--------|
| Blogs marketing signaux payants | Promesses sans méthode ni résultats |
| "AI trading bot" repos sans README | Aucune activité, pas de backing |
