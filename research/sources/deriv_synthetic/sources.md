# 🎰 Synthétiques Deriv — Sources validées

> Sources avec backing sur les instruments synthétiques Deriv (Volatility 75, R_75, R_50, etc.)
> Tags : `data` · `feature` · `model` · `vote` · `risk`

---

## 🟢 Validées avec backing

### Repos GitHub

| # | Nom | ★ | Licence | Tags | Apport | Intégration | Lien |
|---|-----|---|---------|------|--------|-------------|------|
| 1 | **iamMashel/OmashelCap** | ~5 | MIT | `data`, `risk`, `monitor`, `feature` | Bot Deriv production-grade : mock + réel, walk-forward sur vraies bougies Deriv, risk fixed-fractional + circuit breaker, Telegram. **Le plus proche de notre cible.** | Lecture structure + risque (pas copie MIT) | https://github.com/iamMashel/OmashelCap |
| 2 | **deriv-com/deriv-api** | ~76 | MIT | `data` | API officielle WebSocket deriv.app (JS/Node). Référence de vérité pour valider payloads, endpoints, granularités `ticks_history`, `candles`, `ticks`. | Documentation officielle pour notre `data_collector.py` | https://github.com/deriv-com/deriv-api |
| 3 | **0x596173736972/MarketRegimeTrader** | ~18 | MIT | `model`, `feature`, `risk` | HMM + TDA + walk-forward + 30+ métriques. Compatible avec notre stack. | Validation méthode HMM + walk-forward | https://github.com/0x596173736972/MarketRegimeTrading |
| 4 | **alanvito1/ORSTAC** | ~204 | MIT | `feature` (inspiration) | 4 000+ bots XML pour DBot. Dump communautaire — pas une lib Python. Idées stratégies (RSI, moyennes mobiles, martingale). | Lecture stratégies uniquement (pas copie) | https://github.com/alanvito1/ORSTAC |

### Documentation officielle

| # | Ressource | Tags | Apport | Lien |
|---|-----------|------|--------|------|
| 5 | **Deriv API Docs** | `data` | Référence WebSocket officielle : endpoints `ticks_history`, `candles`, `proposal`, `buy`. Granularités, limites de taux, formats. | https://api.deriv.com/ |
| 6 | **Deriv Marketing Article** | `data` | Explications publiques sur les synthétiques : indices volatils générés cryptographiquement, indépendants des marchés réels. | https://blog.deriv.com/ |

### Notes importantes

- **Les synthétiques Deriv (Volatility 75/50, R_75/R_50)** sont des indices générés par des **méthodes stochastiques propriétaires** basées sur le mouvement brownien et la volatilité paramétrique. Ils ne sont pas directement tradables sur des marchés réels — leur comportement statistique est déterministe au niveau de la volatilité cible mais stochastique dans le prix.
- **R_75/R_50** = les versions "R" (rainbow ?) des indices volatils. Même doc API.
- **Volume proxy** : Deriv ne donne pas le volume réel (c'est synthétique). Nous utilisons le nombre de ticks comme proxy (déjà implémenté).
- **Payout fixe** : les options binaires Deriv ont un payout fixe (~85-90%). Le breakeven est de **55.6%** de win rate (soit ~0.556 = 1/1.8).

---

## 🔴 Rejetées

| Ressource | Raison |
|-----------|--------|
| Promesses winrate magique sur synthétiques | Pas de méthode, pas de backtest public |
| Stratégies "martingale" pures | Ruine certaine, pas de backing mathématique |
