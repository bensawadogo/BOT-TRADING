# Système Régime HMM + Fibonacci + Tendance — recherche, architecture, résultats

*8 octobre 2026. Tout est reproductible avec `lab/` (commandes en fin de document).*

## 1. Ce que dit la recherche

| Sujet | Constat | Source |
|---|---|---|
| HMM : forward, forward-backward, Viterbi | Seul le **filtre forward** P(état_t \| x_1..x_t) est causal. Les probabilités **lissées** (forward-backward, `predict_proba` de hmmlearn) et le **chemin Viterbi complet** utilisent le futur. Baum-Welch doit être réestimé en walk-forward, sur le passé seulement. | [Regime Detection Pitfalls](https://r-bloggers.com/2012/11/regime-detection-pitfalls), [Hamilton filter](https://quantvault.org/problem-1067-two-state-volatility-hmm-and-the-hamilton-filter.html) |
| Modèles à sauts | Pénalisent chaque changement d'état, donc des régimes plus stables qu'un HMM. Meilleurs que les HMM classiques quand les régimes sont déséquilibrés et l'historique court (Aydınhan, Kolm, Mulvey, Shu 2024). Paquet `jumpmodels` (Apache-2.0) avec `predict_online` causal. | [Fitting Jump Models](https://arxiv.org/pdf/1711.09220), [Downside risk reduction](https://arxiv.org/pdf/2402.05272), [jumpmodels](https://pypi.org/project/jumpmodels) |
| Fibonacci | Aucune étude académique ne montre d'avantage. Sur 40 243 corrections du forex G10, environ 83 % finissent entre 15 % et 61,8 %, une zone qui contient 4 niveaux de Fibonacci : les « touches » s'expliquent par le hasard. | [forexop](https://forexop.com/strategy/fibonacci-fact-or-fiction/), [HKUST](https://cse.hkust.edu.hk/~rossiter/independent_studies_projects/fibonacci_trading/fibonacci_trading.pdf) |
| Tendance institutionnelle | Le momentum temporel (signe du rendement sur 12 mois, positions dimensionnées par la volatilité, cible de 10 %) est rentable sur plus d'un siècle et sur de nombreux marchés (Moskowitz-Ooi-Pedersen 2012 ; Hurst-Ooi-Pedersen, AQR). | [A Century of Evidence on Trend-Following](https://www.aqr.com/Insights/Research/Journal-Article/A-Century-of-Evidence-on-Trend-Following-Investing) |
| Architecture | Le framework LEAN de QuantConnect, utilisé par des fonds, sépare **Alpha → Construction de portefeuille → Risque → Exécution**, avec des modules interchangeables. | [LEAN](https://www.lean.io/) |

## 2. Ce qui a été repris des projets open source, et ce qui a été corrigé

| Brique | Reprise de | Vérification |
|---|---|---|
| Baum-Welch | `hmmlearn.GaussianHMM.fit` (BSD), 3 initialisations, réestimé tous les 63 jours sur les 1 000 jours précédents | normalisation calculée sur la fenêtre d'entraînement seule |
| Forward | récursion de Rabiner (10 lignes) sur les vraisemblances de hmmlearn | **test** : identique à la dernière ligne de `predict_proba(X[:t+1])` |
| Viterbi | récursion max-produit : dernier état du chemin calculé sur x_1..x_t | **test** : identique à `decode(X[:t+1])[-1]` |
| Modèle à sauts | `jumpmodels.JumpModel` + `predict_online` | **test** de troncature |
| Swings et Fibonacci | concept de `smartmoneyconcepts` (MIT, environ 700 ⭐) | **piège trouvé** : `swing_highs_lows` utilise `shift(-n)`, donc les bougies **futures**. Réécrit en zigzag ATR causal : un sommet n'existe qu'une fois le prix redescendu de 2 × ATR. |
| Construction | volatilité cible de 10 % par marché, poids égal entre marchés, levier ≤ 4 (AQR / MOP) | — |

Le code existant du dépôt (`deriv/ensemble_predictor.py`) entraîne Baum-Welch sur toute la série, validation comprise. Ses anciens backtests étaient donc optimistes.

## 3. Résultats journaliers : 14 marchés, 1985-2026, paramètres fixés avant le test

Forex (10 paires), pétrole WTI et Brent, Nasdaq, gaz naturel. Coûts de 1 à 10 points de base par unité de rotation.

| Variante | Rend./an | Sharpe | t | DD max | 1990-2007 | 2008-2016 | 2017-2026 |
|---|---|---|---|---|---|---|---|
| 1 Momentum 12 mois (institutionnel) | +1,56 % | **0,31** | **2,02** | 31,9 % | 0,71 | 0,00 | −0,34 |
| 2 Tendance 1/3/12 mois | +1,63 % | **0,41** | **2,68** | 27,9 % | 0,74 | 0,09 | −0,34 |
| 3 HMM seul (forward) | −0,36 % | −0,12 | −0,80 | 25,8 % | −0,15 | 0,06 | −0,42 |
| 4 HMM seul (Viterbi) | −0,39 % | −0,13 | −0,86 | 28,0 % | −0,11 | −0,01 | −0,45 |
| 5 Momentum + filtre HMM | +0,52 % | 0,14 | 0,91 | 27,9 % | 0,56 | −0,05 | −0,40 |
| 6 Momentum + filtre à sauts | +1,12 % | 0,29 | 1,89 | **21,7 %** | 0,70 | 0,34 | −0,40 |
| 7 Fibonacci seul | −0,37 % | −0,21 | −1,40 | 23,1 % | −0,57 | 0,06 | −0,11 |
| 8 Fibonacci + momentum + HMM | −0,31 % | −0,26 | −1,68 | 15,0 % | −0,18 | −0,31 | −0,35 |

La volatilité du portefeuille est d'environ 4-5 % parce que les marchés se compensent. Le Sharpe ne dépend pas de ce niveau : on peut monter la mise.

Contrôle hors panier : la tendance sur une **obligation US 10 ans synthétique** (taux FRED, duration de 8 ans) donne un Sharpe de 0,53 à 0,68 (t = 3,4 à 4,3), positif sur les trois sous-périodes. Attention, la baisse séculaire des taux de 1985 à 2020 a aidé les positions acheteuses.

## 4. Résultats H1 Deriv : 10 marchés, environ 9 mois hors échantillon, coûts réels

| Variante | Trades | Réussite | Moy. / trade | t |
|---|---|---|---|---|
| Fibonacci seul | 353 | 55,0 % | −3,9 pb | −0,34 |
| Fibonacci + HMM | 148 | 52,7 % | −1,8 pb | −0,14 |
| Fibonacci + tendance + HMM | 180 | 53,9 % | +1,9 pb | +0,13 |

Aucun marché n'est significatif face au hasard : toutes les valeurs de p sont ≥ 0,17. L'or est positif, à +46 / +95 / +61 pb par trade, mais sur 10 à 29 trades seulement (p de 0,17 à 0,37).

## 5. Conclusion

1. **Ce qui marche, et est documenté depuis un siècle** : la tendance multi-marchés en journalier, avec des positions dimensionnées par la volatilité.
   - Sharpe de 0,3-0,4 sur notre panier de forex et matières premières, plus sur les obligations.
   - Il est faible depuis 2017 sur le forex : c'est connu, ce sont les obligations et les indices qui portent les CTA.
2. **Le HMM (Baum-Welch / Forward / Viterbi) n'apporte rien comme signal de direction.**
   - Utilisé comme filtre de risque, le modèle à sauts réduit le drawdown maximal de 32 % à 22 % pour un rendement presque égal.
   - C'est son vrai rôle chez les institutionnels : savoir quand réduire le risque, pas prédire le sens.
3. **Fibonacci n'a pas d'avantage mesurable**, seul ou combiné, en H1 comme en journalier. C'est conforme à la littérature.
4. **Architecture recommandée pour le bot** :
   - Alpha = tendance 1/3/12 mois sur 10 à 20 marchés ;
   - Construction = volatilité cible ;
   - Risque = modèle à sauts (réduction de l'exposition en régime baissier ou volatil) et coupe-circuit de drawdown ;
   - Exécution = rééquilibrage une fois par jour, au marché.

## 6. Reproduire

```bash
python scripts/fetch_fred.py data/fred
python -m lab.run_systeme daily data/fred
python scripts/fetch_candles_public.py          # bougies H1 Deriv
python -m lab.run_systeme h1 data/frxEURUSD_3600.csv frxEURUSD data/frxXAUUSD_3600.csv frxXAUUSD
pytest lab/tests -q                              # 10 tests, dont 6 de causalité
```
