# Backtest des algorithmes de tendance du dépôt — 08/10/2026

Laboratoire : `lab/` (moteur causal, tests dans `lab/tests/`). Données réelles Deriv
(1 an : 08/10/2025 → 08/10/2026), coûts réels des Multipliers relevés le 08/10/2026.

## Méthode
- Signal à la clôture de la bougie i (données ≤ i), entrée à l'ouverture de i+1.
- Stop et objectif intra-bougie ; si les deux sont touchés : stop (prudent).
- HMM réentraîné sur le passé uniquement (`lab/regime.py`), aucun pickle.
- Chaque résultat est comparé à des entrées AU HASARD avec les mêmes sorties.
- 70 % de l'historique = réglage, 30 % les plus récents = test.
- Indicateurs Pine sans règle de sortie : stop 1,5 ATR / objectif 2,5 ATR (celle de B).

## Algorithmes testés
| Code | Source |
|---|---|
| A6 / A5 | `tradingview/hmm_kalman_consensus_v6.pine` / `indicator.pine` |
| B | `deriv/strategies/regime_momentum_strategy.py` (classe d'origine) |
| B2 | B corrigé (voir défaut ci-dessous) |
| C | `intelligence/signal_engine.py` + seuils de `rise_fall.py` |
| D | `crypto/strategies/HMMRegimeStrategy.py` |

## Résultats
- **M5, milliers de trades** (R_75 100 000 bougies, or 69 575) : tous négatifs après
  coûts, aucun ne bat le hasard de façon stable (ex. C sur R_75 : 5 203 trades, −4 bp/trade).
- **H1, 8 paires forex mises en commun** (coûts déduits) :

| Algo | Trades | Réussite | Moyenne / trade |
|---|---|---|---|
| A6 | 214 | 39,7 % | −1,3 bp |
| A5 | 85 | 47,0 % | +2,7 bp (non significatif) |
| B2 | 274 | 37,6 % | −3,3 bp |
| C | 1 762 | 39,0 % | −2,4 bp |
| D | 1 168 | 38,1 % | −3,6 bp |

- **Or H1/H4, R_75 H1** : mêmes conclusions. Les quelques résultats « p ≤ 0,05 »
  portent sur 2 à 8 trades ou s'inversent d'une période à l'autre (ex. D USDJPY :
  +20 bp en réglage, −20 bp en test) : sur ~150 tests, quelques faux positifs sont attendus.
- **B2 sur EUR/USD H1** (+13,6 / +5,5 bp, 14 trades) ne se retrouve sur aucune
  autre paire avec la même règle → hasard.

## Défauts trouvés dans le code d'origine
1. **B ne déclenche presque jamais** : la machine à états se réinitialise dès que la
   confiance HMM passe sous 70 % ou que le régime quitte Bull — ce qui arrive
   précisément pendant le pullback attendu (0 signal sur l'or H4 malgré 347 bougies
   remplissant les conditions d'armement). Corrigé dans `lab` (B2), sans edge.
2. **D (freqtrade) a un biais de look-ahead** : le HMM est ajusté sur toute la série
   (`hmm.fit(prices)` dans `populate_indicators`) puis utilisé pour dater les régimes
   passés. En backtest freqtrade, les régimes « connaissent » le futur.
3. **C** utilise un modèle HMM pré-entraîné chargé depuis un pickle, quelle que soit
   la donnée : son régime n'est pas celui du marché tradé.

## Conclusion
Aucun des algorithmes du dossier n'a d'edge mesurable après coûts sur l'or, 8 paires
forex et Volatility 75, en M5, H1 et H4. Ne pas les trader en réel tels quels.
