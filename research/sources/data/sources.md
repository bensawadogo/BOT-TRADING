# 📦 Sources de données — Datasets & APIs

> Inventaire des flux et datasets disponibles pour enrichir notre data collector.

---

## Données temps réel (WebSocket / API)

| Source | Couverture | Latence | Prix | Fiabilité | Usage |
|--------|-----------|---------|------|-----------|-------|
| **Deriv WebSocket** (officiel) | Synthétiques + forex/crypto | ~100ms | Gratuit (token) | ✅ Haute | Flux principal ticks/bougies |
| **Binance WebSocket** | Crypto spot + futures | ~50ms | Gratuit | ✅ Haute | Backup crypto (BTC, ETH) |
| **Polygon.io** | Forex + crypto tick | ~200ms | Freemium | ✅ Haute | Backup forex premium |
| **Twelve Data** | Forex + crypto | ~300ms | Freemium | ✅ Haute | Alternative multi-assets |

## Données historiques (batch)

| Source | Couverture | Format | Prix | Usage |
|--------|-----------|--------|------|-------|
| **Deriv ticks_history** | Synthétiques + forex/crypto | JSON via WS | Gratuit | Backtest initial |
| **Dukascopy** | Forex + crypto (depuis 2000) | CSV | Gratuit | Backtest long forex |
| **Binance aggTrades** | Crypto spot (par pair) | CSV/JSON | Gratuit | Backtest crypto long |
| **FRED / ECB** | Macro US/EU | CSV/JSON | Gratuit | Features contextuelles (optionnel) |

## Données académiques (backtesting)

| Source | Couverture | Format | Usage |
|--------|-----------|--------|-------|
| **AlgoSeek SP500** (via ML4T) | US equities daily | Parquet | Benchmark features XGBoost |
| **Kaggle forex/crypto** | Variable | CSV | Exploration |

## Caveats & pièges

- **Synthétiques Deriv** : pas de volume réel, séries limitées en temps (l'historique via API est souvent limité à quelques milliers de ticks).
- **Forex/Crypto** : le spread et les requotes ne sont pas toujours reflétés dans les données OHLCV "mid-price". Le coût réel doit être estimé (~0.1-0.5 pips forex, ~0.05% crypto).
- **Surajustement** : plus de données ≠ meilleur modèle. Toujours valider out-of-sample (walk-forward).
