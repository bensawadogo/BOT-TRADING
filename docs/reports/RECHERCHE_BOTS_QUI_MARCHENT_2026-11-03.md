# Les bots qui marchent : ce qu'ils font, ce qu'on peut leur prendre (03/11/2026)

*Recherche ciblée : qui gagne vraiment avec des systèmes automatiques, comment, et ce qui
est transposable à ce projet avec ses contraintes réelles (Burkina Faso, capital limité,
Deriv MT5). Aucune ligne de code modifiée : ce document sert à décider.*

## 1. Qui gagne vraiment, et comment

| Type | Exemples | Pourquoi ça marche | Accessible ici ? |
|---|---|---|---|
| Teneurs de marché, haute fréquence | Citadel, Virtu, Jump | Latence, colocation, rabais de frais, flux d'ordres | **Non** |
| Fonds de tendance (CTA) | Man AHL, Winton, AQR | 50 à 100+ marchés à terme, coûts minimes, discipline | **Oui, en version réduite** : c'est notre bot |
| Portage (carry) | Fonds macro ; en crypto : cash-and-carry, Ethena | Être payé pour détenir (différentiel de taux, prime des contrats perpétuels) | Partiellement |
| Bots vendus aux particuliers | Grid, DCA, « IA », groupes de signaux | Rarement documenté ; souvent sur-ajusté | À éviter |

### Les fonds de tendance : notre modèle
- Notre bot reprend leur alpha (tendance multi-horizons, volatilité cible). La différence
  est dans **l'exécution** : ils tradent des contrats à terme (futures) sur des dizaines de
  marchés peu corrélés (taux, obligations, matières premières, indices, devises), sans la
  marge de financement des CFD.
- Même pour eux, les années sont irrégulières : SG Trend Index +27,3 % en 2022, puis des
  années difficiles (encore −2 % sur le seul mois de mai 2025).
- **Rob Carver** (ex-Man AHL) est la référence la plus utile pour nous. Il trade son propre
  argent en futures, publie sa méthode (*Systematic Trading*, *Leveraged Trading*,
  *Advanced Futures Trading Strategies*) et son moteur open source, **pysystemtrade**
  (connecteur Interactive Brokers).

### Le portage crypto (cash-and-carry)
- Le principe : acheter au comptant et vendre le contrat à terme (ou le perpétuel). On
  encaisse l'écart, sans parier sur la direction.
- La BIS (document de travail 1087) mesure un portage crypto **supérieur à 10 %/an en
  moyenne**, avec des krachs.
- Depuis 2024, l'écart est comprimé par l'afflux de capital d'arbitrage (Ethena, entre
  autres) : le financement des perpétuels tend vers 0,01 % par 8 h, soit ≈ 11 %/an brut
  en période haussière, et beaucoup moins sinon.
- Risques :
  - faillite de la plateforme (FTX en 2022) ;
  - liquidation de la jambe vendeuse ;
  - financement négatif ;
  - statut légal flou dans l'UEMOA ;
  - accès depuis le Burkina non vérifié (Binance P2P n'est confirmé par aucune source fiable).

### Ce que les bots vendus aux particuliers ne disent pas
- **Market making** (Hummingbot) : la documentation du projet elle-même dit qu'il gagne en
  marché plat, perd en tendance (risque d'inventaire), et que la simulation surestime les
  gains. Un utilisateur interrogé par le projet se dit « à l'équilibre, peut-être
  légèrement positif ».
- **Bots grid** : même pari que le market making (le marché revient), même perte en tendance.
- **Particuliers en général** : 74 à 89 % des comptes CFD perdent (ESMA). Parmi les day
  traders brésiliens tenaces, 1,1 % gagnent plus que le salaire minimum.
- **Frameworks** (Freqtrade, NautilusTrader, QuantConnect) : ils ne rapportent rien par
  eux-mêmes. Leur valeur est l'ingénierie : même code en backtest et en live, détection
  du look-ahead (Freqtrade `lookahead-analysis`, `recursive-analysis`), simulation avant
  le réel. Le projet applique déjà le principe « même code » et teste la causalité.

## 2. Ce que les pros font et que notre bot ne fait pas encore

| Pratique des pros | Notre bot aujourd'hui | Amélioration possible |
|---|---|---|
| **Instruments sans marge de financement** (futures) | CFD Deriv : swaps de 1,3 à 2,9 %/an sur le forex, 60-75 % sur le pétrole | Compte **sans swap** Deriv (39 paires forex + pétrole), puis futures |
| **Diversification** : 45 à 100+ marchés | 11 marchés, dont 10 paires forex toutes liées au dollar | Pétrole (sans swap), or et indices si éligibles, puis taux et matières premières en futures |
| **Prévisions continues** : croisements EWMAC 16/64, 32/128, 64/256, divisés par la volatilité, plafonnés à ±20, combinés | Signe des rendements à 1/3/12 mois (−1, −⅓, +⅓, +1) | Positions plus progressives, moins de rotation |
| **Deuxième source de rendement** : portage (Carver : 60 % tendance, 40 % portage) | Tendance seule | Portage via le roll des futures ; inutile sur un compte sans swap (aucun taux payé ni reçu) |
| **Optimisation dynamique** pour petit capital | Arrondi marché par marché : 9 marchés sur 16 tombaient à 0 lot | Choisir les lots entiers qui suivent le mieux le portefeuille idéal (voir ci-dessous) |
| **Budget de coûts** par marché | Aucun filtre | Exclure un marché si ses coûts annuels mangent une trop grande part de son rendement attendu |
| **Suivi public live vs backtest** | Démo jamais lancée | Rapport mensuel : équité de la démo face au backtest sur la même période |

### Le point clé pour un petit compte : l'optimisation dynamique de Carver
- **Le problème.** Avec peu de capital, arrondir chaque marché au lot entier détruit le
  portefeuille. C'est exactement ce que fait notre bot aujourd'hui.
- **Sa solution.** Il part de zéro et ajoute, un lot à la fois, celui qui rapproche le plus
  le portefeuille réel du portefeuille idéal. La distance est mesurée par l'erreur de suivi
  calculée avec la covariance des marchés, plus une pénalité de coûts.
- **Résultats publiés, 45 instruments :**
  - à 100 000 $ : Sharpe 1,06 en arrondi simple, 1,19 optimisé, pour 1,17 en idéal non arrondi ;
  - à 50 000 $ : optimisé ≈ +0,1 de Sharpe par rapport à l'arrondi, mais −0,2 par rapport à l'idéal ;
  - il ne détient en moyenne que 6 instruments sur 45 à la fois.
- **Ce que ça veut dire pour nous.** La méthode atténue le problème du capital, elle ne le
  supprime pas. La version d'octobre 2021 avait un défaut de calcul des coûts, que Carver a
  corrigé en novembre 2021 : c'est la version corrigée qu'il utilise en production.

## 3. Trois chemins, avec leurs vrais chiffres

### Chemin 1 — Rester sur Deriv MT5 avec un compte sans swap (tout de suite, gratuit)
- **Ce qu'il apporte.** Plus de swap, et le pétrole redevient tradable. La compatibilité
  avec le Burkina est déjà prouvée.
- **Le prix.** Le spread est de 2,2 pips au départ, contre 0,6 sur le compte standard,
  soit environ 1 bp par unité de rotation. C'est justement l'hypothèse de coût du backtest
  d'octobre, qui donnait ≈ 4,5 %/an (Sharpe 0,54). Ce chiffre reste à vérifier sur le
  compte réel.
- **À vérifier :**
  - la liste exacte des marchés éligibles (indices ? or ?) ;
  - un éventuel plafond de durée de détention ;
  - les commissions.

### Chemin 2 — Contrats à terme via Interactive Brokers (le chemin des pros)
- IBKR accepte les résidents du Burkina Faso d'après sa liste officielle de pays.
- Les micro-futures font 1/10 des contrats standards : micro pétrole (100 barils), micro or
  (10 onces), micro indices, micro devises.
- **Ce qu'il apporte :**
  - pas de marge de financement CFD ;
  - vraie diversification, avec les taux et les obligations ;
  - pysystemtrade, déjà connecté à IBKR.
- **Le prix :**
  - le capital : les résultats de Carver sont mesurés à 50 000-100 000 $ ;
  - un virement international pour alimenter le compte ;
  - plus de complexité.

### Chemin 3 — Portage crypto neutre au marché (recherche séparée)
- C'est le seul « bot qui gagne » documenté et accessible à un particulier.
- Mais le rendement est comprimé depuis 2024, il y a un risque de plateforme, et le cadre
  légal et l'accès restent à vérifier.
- Il faudrait un connecteur crypto et une surveillance permanente.

**Recommandation : chemin 1 maintenant, en améliorant le bot avec les méthodes des pros ;
chemin 2 quand le capital le permettra ; chemin 3 seulement comme recherche à part.**

## 4. Plan proposé (à valider avant tout code)
1. **Mesurer le compte sans swap (aucun code).** Ouvrir un compte démo sans swap. Relancer
   `scripts/couts_mt5.py tendance` dessus, puis le backtest avec ces coûts. Décision
   chiffrée : quels marchés, quel rendement attendu.
2. **Prévisions continues EWMAC** à la place des signes.
   - Protocole : réglage choisi sur 1985-2007, vérifié sur 2008-2026.
   - On l'adopte seulement si la période de vérification s'améliore.
   - Tests : causalité et « live = backtest ».
3. **Optimisation dynamique** des lots, mesurée à 5 000, 10 000 et 25 000 $ avec les vrais
   lots minimums de Deriv.
4. **Budget de coûts** par marché, et **diversification** (pétrole sur le compte sans
   swap, or et indices s'ils sont éligibles) : chaque ajout backtesté avec ses coûts.
5. **Rapport mensuel live vs backtest** dans le dashboard.
6. **Plus tard :** micro-futures IBKR, en reprenant ce qui marche dans pysystemtrade.

À ne pas copier : l'optimisation de dizaines de paramètres (sur-ajustement), les bots grid,
les « IA » boîte noire, le scalping (`ANALYSE_SCALPING_2026-11-02.md`).

## Sources
- Carver, optimisation dynamique : [Mr Greedy and the Tale of the Minimum Tracking Error Variance](https://qoppac.blogspot.com/2021/10/mr-greedy-and-tale-of-minimum-tracking.html) ; [entretien sur son système de futures](https://thealgorithmicadvantage.substack.com/p/033-rob-carver-the-comprehensive)
- Tendance + portage à la Carver (réimplémentation) : [QuantConnect, Combined Carry and Trend](https://www.quantconnect.com/research/16001/combined-carry-and-trend/)
- Portage, toutes classes d'actifs : [Koijen, Moskowitz, Pedersen, Vrugt, *Carry* (NBER 19325)](https://www.nber.org/papers/19325)
- Portage crypto : [BIS, document de travail 1087 *Crypto carry*](https://www.bis.org/publ/work1087.htm) ; [BitMEX, structure des taux de financement, T3 2025](https://www.bitmex.com/blog/2025q3-derivatives-report)
- Fonds de tendance : [Hedgeweek, année 2022](https://www.hedgeweek.com/trend-followers-turn-leaders-ctas-deliver-record-returns-2022/) ; [Alternatives Watch, mai 2025](https://www.alternativeswatch.com/2025/06/02/trend-following-hedge-funds-endure-more-pain-societe-generale-indices/)
- Deriv sans swap : [types de comptes](https://deriv.com/ae/help-centre-question/what-types-of-trading-accounts-does-deriv-offer) ; [guide du compte sans swap](https://deriv.com/ae/academy/trading-guides/swap-free-trading-deriv-mt5)
- Interactive Brokers : [pays acceptés](https://www.interactivebrokers.co.uk/en/index.php?f=38309) ; CME : [micro WTI](https://cmegroup.com/education/articles-and-reports/micro-wti-crude-oil-futures-faq.html)
- Market making : [Hummingbot, idées reçues](https://docs.hummingbot.org/academy-content/top-beginner-misconceptions-about-market-making/) ; [risque d'inventaire](https://hummingbot.org/blog/what-is-inventory-risk/)
- Freqtrade : [lookahead-analysis](https://www.freqtrade.io/en/stable/lookahead-analysis/) ; [recursive-analysis](https://www.freqtrade.io/en/stable/recursive-analysis)
- Particuliers : [Quantpedia, day trading au Brésil](https://quantpedia.com/retail-day-trading-is-an-uphill-battle/) ; [ESMA, mesures CFD 2018](https://rpclegal.com/thinking/financial-services-regulatory-and-risk/esma-cracks-down-on-the-sale-of-binary-options-and-cfds-to-retail-investors)
