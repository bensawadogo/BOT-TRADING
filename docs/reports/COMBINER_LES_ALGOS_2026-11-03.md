# Faire travailler ensemble les trois algos de tendance (03/11/2026)

*Recherche, sans code. Les trois algos du projet : la **tendance** (momentum 1/3/12 mois),
le **régime** (HMM / modèle à sauts) et **Fibonacci** (entrée sur retracement). Question :
comment les combiner pour qu'ils s'aident au lieu de se gêner ?*

## 1. Ce que le projet a déjà mesuré (1985-2026, 14 marchés, journalier)

| Algo | Seul | Combiné |
|---|---|---|
| Tendance 1/3/12 mois | Sharpe 0,41 à 0,44 | — |
| Régime (modèle à sauts) | −0,12 comme signal de sens | **En filtre de la tendance : Sharpe 0,57, drawdown 42 % → 28 %** |
| Fibonacci | −0,21 | Fibonacci + tendance + HMM : −0,26 (journalier), +1,9 bp/trade, t = 0,13 (H1) |

Leçon : chaque algo n'est utile que dans **son** rôle. Le régime aide comme gestion du
risque, pas comme prédicteur. Fibonacci n'a d'avantage nulle part. Et exiger que les trois
soient d'accord (le « vote » de l'ancien bot Deriv) a toujours dégradé les résultats.

## 2. Ce que dit la recherche sur la combinaison de signaux

### 2.1 Moyenner, ne pas voter ; poids simples, pas optimisés
- **Le « puzzle de la combinaison de prévisions »** (Stock et Watson 2004) : la simple
  moyenne de plusieurs prévisions bat, de façon répétée, les poids « optimaux » estimés.
  Les poids estimés portent une erreur d'estimation et sont instables
  (Smith-Wallis 2009, Claeskens et al.).
- **DeMiguel, Garlappi, Uppal (2009)** : aucun des 14 modèles d'optimisation testés ne bat
  régulièrement la répartition 1/N hors échantillon. Il faudrait environ 3 000 mois de
  données pour que l'optimisation l'emporte avec 25 actifs.
- **Carver** (*Systematic Trading*) : on additionne des **prévisions continues** avec des
  poids presque égaux. On corrige par un « multiplicateur de diversification » (la moyenne
  de signaux peu corrélés est plus faible que chaque signal) et on plafonne à ±20.
- Conséquence pour nous : la règle « au moins 4 sur 7 d'accord » jette l'information. Une
  moyenne pondérée de signaux gradués la garde.

### 2.2 Mélanger plusieurs vitesses de tendance
- **Man AHL, « The Need for Speed »** :
  - les tendances rapides et lentes gagnent toutes à long terme, mais sont **peu corrélées**
    (0,17 entre la plus rapide et la plus lente, 1995-2022) ;
  - les rapides coûtent plus cher en frais, mais protègent mieux dans les krachs ;
  - aucune vitesse n'est parfaite, ils recommandent de les mélanger.
- **Goulding, Harvey, Mazzoleni, *Momentum Turning Points* (JFE 2023)** :
  - le point faible de la tendance, ce sont les **retournements** : le signal lent réagit
    trop tard, le rapide donne de fausses alertes ;
  - quand lent et rapide ne sont pas d'accord (« correction » ou « rebond », 35 % du
    temps), c'est un retournement probable ;
  - les stratégies de vitesse intermédiaire, obtenues en mélangeant lent et rapide, ont un
    Sharpe **supérieur à la moyenne** de celui du lent et du rapide.
- **Version apprentissage profond** : Wood et al., *Slow Momentum with Fast Reversion*
  (JFDS 2022). Ajouter un détecteur de rupture à un réseau de tendance a amélioré le Sharpe
  d'environ un tiers sur 50 contrats à terme (1995-2020). Lim, Zohren et Roberts (2019)
  annoncent un doublement hors frais. C'est prometteur, mais lourd, gourmand en données et
  facile à sur-ajuster.

### 2.3 Le régime comme bouton de risque, pas comme boussole
- **Shu, Yu, Mulvey (2024)**, les auteurs du paquet `jumpmodels` que le bot utilise :
  réduire l'exposition dans les régimes défavorables détectés par un modèle à sauts réduit
  nettement la volatilité et le drawdown maximal. C'est exactement ce que notre filtre fait,
  et ce que nos chiffres confirment.
- **Gestion par la volatilité** :
  - Barroso et Santa-Clara (2015) : dimensionner le momentum actions par sa volatilité fait
    passer le Sharpe de 0,53 à 0,97 et évite la plupart des krachs ;
  - mais Cederburg et al. (2020) trouvent que, hors échantillon, sur 103 stratégies, la
    version gérée par la volatilité ne gagne que 53 fois contre 50 ;
  - notre bot cible déjà la volatilité : ne pas empiler une deuxième couche du même type.
- **Coupe-circuit** : selon Kaminski et Lo (2014), un stop réduit le rendement si les prix
  sont aléatoires, mais ajoute de la valeur en présence de momentum. Il est donc cohérent
  avec une stratégie de tendance.

### 2.4 Une seule position par marché (intégrer plutôt que juxtaposer)
- AQR (Fitzgibbons et al. 2016) : additionner les scores des signaux AVANT de construire
  le portefeuille fait mieux que de faire tourner des sous-portefeuilles séparés. Ils
  annoncent +1 %/an en portefeuille long-only contraint. Des contre-études nuancent
  (Newfound, Alpha Architect).
- Pour nous, l'argument décisif est le coût : des signaux qui se contredisent dans des
  sous-portefeuilles séparés génèrent des ordres qui s'annulent et paient deux fois le
  spread. Une position nette par marché, c'est un seul ordre.

### 2.5 Le méta-labeling : pour plus tard
- Le principe (López de Prado, Joubert 2022) : le signal principal (la tendance) donne le
  sens, un second modèle apprend **quand le croire et combien miser**, à partir du contexte
  (régime, volatilité, position dans le retracement…).
- Le but : filtrer les faux signaux et améliorer Sharpe et drawdown.
- C'est le seul cadre rigoureux où Fibonacci pourrait servir : comme **variable de
  contexte**, jamais comme feu vert. Mais avec des données journalières sur une dizaine de
  marchés, le risque de sur-ajustement est élevé.

### 2.6 Le piège : tester trop de combinaisons
- Chaque combinaison essayée est un « essai ». Le **ratio de Sharpe dégonflé**
  (Bailey et López de Prado, 2014) corrige le meilleur résultat selon le nombre d'essais.
- La **probabilité de sur-ajustement du backtest** (Bailey et al.) vérifie que le
  « meilleur » le reste hors échantillon.
- Sans ces garde-fous, tester 50 combinaisons « trouve » toujours un gagnant par hasard.

## 3. L'architecture recommandée : un rôle par algo

```text
            ┌─────────────────────────────────────────────────────────┐
 Couche 1   │ DIRECTION  — tendance à 3-4 vitesses (prévisions          │
            │              continues, moyenne à poids égaux, plafond)   │
            └───────────────────────┬─────────────────────────────────┘
 Couche 2   │ CONFIANCE  — accord lent / rapide (Goulding et al.) :     │
            │              exposition réduite en « correction/rebond »  │
            └───────────────────────┬─────────────────────────────────┘
 Couche 3   │ RISQUE     — régime (modèle à sauts) : réduit ou coupe ;  │
            │              volatilité cible ; coupe-circuit             │
            └───────────────────────┬─────────────────────────────────┘
 Exécution  │ UNE position nette par marché ; lots par optimisation     │
            │ dynamique ; zone neutre                                   │
            └─────────────────────────────────────────────────────────┘
 Fibonacci  : retiré des décisions. Éventuellement variable de contexte dans un
              méta-labeling futur, ou test de « timing d'entrée » isolé.
```

Ce qui change par rapport au bot actuel :
1. La tendance passe de 3 signes (−1, −⅓, +⅓, +1) à des prévisions graduées sur plusieurs
   vitesses : moins de rotation, transitions plus douces.
2. Le « troisième algo » n'est plus Fibonacci, sans preuve, mais le **désaccord entre
   tendance lente et rapide**, documenté dans le JFE.
3. Le régime reste un bouton de risque. À tester : réduire de moitié au lieu de couper.

## 4. Protocole d'essai (à valider avant tout code)

Variantes fixées **à l'avance**, réglées sur 1985-2007, vérifiées sur 2008-2026, aux coûts
du compte réellement utilisé (sans swap si ouvert) :

| # | Variante | Ce qu'on vérifie |
|---|---|---|
| E0 | Bot actuel | Référence |
| E1 | Tendance continue, 3 vitesses (EWMAC 16/64, 32/128, 64/256) | Moins de rotation, Sharpe ≥ E0 |
| E2 | E1 + exposition réduite si lent et rapide se contredisent | Moins de pertes aux retournements |
| E3 | E2 avec régime « réduire de moitié » au lieu de « couper » | Rendement contre drawdown |
| E4 | E0 + Fibonacci en timing d'entrée seulement | Attendu : aucun gain (contrôle) |

Règles d'adoption :
- on garde une variante seulement si elle améliore la période **2008-2026** ;
- son Sharpe dégonflé (5 essais) doit rester significatif ;
- le test « live = backtest » doit passer.

Cinq essais, pas cinquante.

## Sources
- Prévisions : [puzzle de la combinaison (Elliott, UCSD)](https://econweb.ucsd.edu/~gelliott/Combination.pdf) ; [On the forecast combination puzzle](https://experts.umn.edu/en/publications/on-the-forecast-combination-puzzle/)
- 1/N : [DeMiguel, Garlappi, Uppal (RFS 2009)](https://ideas.repec.org/a/oup/rfinst/v22y2009i5p1915-1953.html)
- Vitesses : [Man AHL, The Need for Speed](https://www.man.com/insights/need-for-speed-trend-following) ; [Institutional Investor](https://www.institutionalinvestor.com/article/b1h96m4kh5lq0w/how-to-make-one-of-the-best-performing-hedge-fund-strategies-even-better)
- Retournements : [Momentum Turning Points (Duke)](https://scholars.duke.edu/publication/1586630) ; [CXO Advisory](https://www.cxoadvisory.com/technical-trading/mitigating-impact-of-price-turning-points-on-trend-following/) ; [Breaking Bad Trends (IBKR)](https://www.interactivebrokers.com/campus/ibkr-quant-news/breaking-bad-momentum-trends/)
- Apprentissage profond : [Slow Momentum with Fast Reversion](https://arxiv.org/pdf/2105.13727) ; [Deep Momentum Networks](https://arxiv.org/abs/1904.04912)
- Régime : [Shu, Yu, Mulvey 2024](https://arxiv.org/abs/2402.05272v3)
- Volatilité : [Momentum has its moments](https://ciencia.ucp.pt/en/publications/momentum-has-its-moments/) ; [Cederburg et al. 2020](https://www.lehigh.edu/~xuy219/research/COWY.pdf)
- Stops : [Kaminski et Lo (MIT)](https://dspace.mit.edu/handle/1721.1/114876)
- Intégration des signaux : [AQR, Don't Just Mix, Integrate](https://www.aqr.com/insights/research/white-papers/long-only-style-investing) ; [contre-point Alpha Architect](https://alphaarchitect.com/should-investors-combine-or-separate-their-factor-exposures/)
- Méta-labeling : [Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling) ; [Hudson & Thames](https://hudsonthames.org/tag/meta-labeling/)
- Sur-ajustement : [Deflated Sharpe Ratio](https://papers.ssrn.com/abstract=2460551) ; [PBO](https://archive.linux.duke.edu/cran/web/packages/pbo/vignettes/pbo.html)
- Carver : [Systematic Trading (Harriman)](https://harriman-house.com/systematic-trading) ; [QuantConnect, tendance + portage](https://www.quantconnect.com/research/16001/combined-carry-and-trend/)
