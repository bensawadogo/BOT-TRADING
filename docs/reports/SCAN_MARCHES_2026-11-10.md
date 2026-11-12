# Balayage de 124 marchés Deriv MT5 : la tendance n'a d'avantage nulle part (10/11/2026)

*Question posée : les cryptos, les actions, les indices, les métaux, les ETF, les matières
agricoles et les paires de devises croisées ont-ils un avantage de tendance que le panier
forex n'a pas ? Réponse : non. Script : `scripts/scan_marches.py` (fetch, couts, run, classes) ;
données brutes : `data/scan_marches.csv`, `data/scan_classes.pkl`.*

## 1. Méthode

- **Stratégie** : le bot d'origine (E0, signes 1/3/12 mois, volatilité cible 10 %, zone neutre),
  appliqué **tel quel** à chaque marché. Rien n'a été réglé sur ces données. Filtre de régime
  désactivé pour le balayage (il faudrait des heures ; il est testé séparément sur le panier
  de production).
- **Historique** : clôtures journalières Yahoo (proxy des CFD Deriv, qui n'ont que 2 à 8 ans
  d'historique), depuis 1995 au plus tôt. Lundi-vendredi seulement (règle du projet).
- **Coûts réels Deriv MT5**, relevés dans le terminal le 07/11/2026 : demi-spread médian des
  ticks à l'heure d'ouverture du marché, swaps long et court convertis en fraction annuelle,
  lot minimum. Deux conventions : **A** = celle du projet (spread + surcoût de swap
  = −(swap long + swap court)/2) ; **B** = swaps réels payés côté par côté.
- **Mesures** : Sharpe net, et surtout l'**alpha** : rendement du bot non expliqué par le marché
  lui-même acheté et gardé (régression), avec son t de Student. Un marché qui monte donne un
  Sharpe positif à n'importe quelle règle acheteuse ; l'alpha dit ce que la tendance ajoute.
- **Référence pour le hasard** : sur 124 essais indépendants sans aucun avantage, on attend
  ≈ 3 marchés au-dessus de t = 2 et ≈ 3 en dessous de t = −2.

## 2. Résultat marché par marché

| Classe | N | Années (méd.) | Sharpe méd. | Part Sharpe > 0 | Alpha méd. | t alpha méd. | t > 2 | t < −2 |
|---|---|---|---|---|---|---|---|---|
| Indices boursiers | 15 | 31 | −0,01 | 47 % | 0,0 % | +0,2 | 0 | 0 |
| Crypto | 25 | 8 | +0,19 | 60 % | 0,0 % | 0,0 | 1 | 3 |
| Actions US/DE | 40 | 31 | −0,12 | 20 % | −1 % | −0,7 | 1 | 10 |
| Forex croisés | 16 | 20 | −0,17 | 19 % | −2 % | −0,9 | 0 | 3 |
| Métaux | 5 | 25 | −0,14 | 20 % | −2 % | −1,1 | 0 | 2 |
| ETF | 19 | 22 | −0,45 | 5 % | −3 % | −2,0 | 0 | 10 |
| Agricoles | 4 | 26 | −0,40 | 0 % | −4 % | −2,5 | 0 | 3 |
| **Total** | **124** | | | **28 %** | | **−0,8** | **2** | **31** |

- **2 marchés sur 124 dépassent t = 2** (NVDA +0,58, t = 2,3 ; SOL +0,86, t = 2,0) : exactement
  ce que le hasard produit. **31 sont sous t = −2** : dix fois plus que le hasard. Le bot ne
  fait pas « rien » sur ces marchés, il **perd**.
- **Pourquoi il perd** : les swaps. Les ETF obligataires (AGG, LQD, HYG) ont un swap vendeur de
  −17 à −23 %/an ; leur volatilité est faible (4-8 %), donc la cible de 10 % les fait trader
  à levier 2 à 3 : Sharpe −2,0 à −2,2, t = −10. Même mécanique pour le cuivre (swap acheteur
  −56 %/an), le sucre (−23 % des deux côtés), USDTHB (−17 à −22 %). **Un marché peu volatil
  avec un swap élevé est intradable en tendance chez Deriv, quelle que soit la règle.**
- **Crypto** : le Sharpe médian positif (+0,19) est la hausse du marché, pas la tendance :
  l'alpha médian est 0,0 %. Et le swap est de −15 à −25 %/an **dans les deux sens**.
- **Actions** : Deriv ne liste que les grandes valeurs d'aujourd'hui (biais du survivant en
  faveur des résultats) ; malgré cela, 80 % sont négatives. NVDA en tête est le meilleur tirage
  sur 124, pas une découverte.
- **Indices** : la seule classe à zéro plutôt que négative (DAX t = 2,0, AEX t = 1,5, les 13
  autres rien). Indice vs ETF du même marché : SPX +0,03 / SPY −0,28, uniquement par le swap
  (−6 % / +2 % sur l'indice contre −2 % / −8 % sur l'ETF).

## 3. Résultat en portefeuille, par classe

Chaque classe tradée en panier entier (convention A, sans régime), contre le panier forex de
production (11 marchés) **sur la même période** ; P = part des rééchantillonnages par blocs où
la classe bat le forex (0,5 = indiscernable).

| Panier | Période | Sharpe panier | Sharpe forex 11 (même période) | CAGR | DD max | P(classe > forex) |
|---|---|---|---|---|---|---|
| Forex 11 (production, sans régime) | 1996-2026 | −0,04 | | −0,6 % | 42 % | |
| Indices (15) | 1996-2026 | −0,13 | −0,04 | −1,4 % | 63 % | 0,29 |
| **Crypto (25)** | 2015-2026 | **+0,58** | −0,44 | +4,7 % | 28 % | **0,99** |
| Forex croisés (16) | 2000-2026 | −0,60 | −0,15 | −4,5 % | 74 % | 0,00 |
| Actions (40) | 1996-2026 | −0,31 | −0,04 | −2,8 % | 65 % | 0,10 |
| ETF (19) | 1996-2026 | −1,01 | −0,04 | −7,5 % | 92 % | 0,00 |
| Métaux (5) | 1998-2026 | −0,33 | −0,17 | −2,9 % | 71 % | 0,27 |
| Forex 11 + indices | 1996-2026 | +0,07 | −0,04 | +0,2 % | 47 % | 0,73 |

**Le seul panier qui bat le forex est la crypto.** Mais le test qui compte est le suivant :
acheter et garder les mêmes 25 cryptos, aux mêmes poids de risque, avec les mêmes swaps,
ramené à la même volatilité :

| 2015-2026 | Sharpe |
|---|---|
| Tendance (le bot) sur 25 cryptos | 0,58 |
| **Acheter et garder** les mêmes cryptos | **0,84** |
| Alpha de la tendance sur acheter-et-garder | +2,5 %/an, **t = 1,0** (bêta 0,34) |
| 2015-2020 : tendance / acheter-et-garder | 0,95 / 1,17 |
| 2021-2026 : tendance / acheter-et-garder | 0,13 / 0,49 |

Le bot sur la crypto gagne **parce que la crypto a monté**, et il gagne **moins** que la
position acheteuse passive. La tendance n'ajoute rien de mesurable (t = 1,0) et la période
couvre un seul cycle haussier. Ajouter les cryptos au bot, c'est parier sur la poursuite de la
hausse en payant −15 à −25 %/an de swap pour le faire.

Ajouter les 15 indices au panier forex : +0,07 contre −0,04, P = 0,73, indiscernable.

## 4. Conclusion

1. **Aucun marché, aucune classe n'a d'avantage de tendance mesurable aux coûts de Deriv.**
   Le résultat est pire que neutre : 31 marchés sur 124 perdent de façon significative, à cause
   des swaps sur les instruments peu volatils.
2. **Le panier du bot ne change pas.** Rien de mesuré ne justifie d'y ajouter un marché.
   Les cryptos ne sont pas ajoutées : leur Sharpe est du bêta, et acheter-et-garder fait mieux.
3. **Le problème est le coût de détention chez ce courtier, pas le signal** : conclusion déjà
   tirée le 06/11 (`COMPTE_SANS_SWAP_2026-11-06.md`), confirmée ici sur 124 marchés.
4. Note : le panier forex **sans régime et depuis 1996** fait −0,04. Les 0,33 du rapport du
   02/11 viennent du filtre de régime et de la décennie 1985-1995. Le bot en production garde
   le régime ; sur 2008-2026 il est à 0,10 (`ESSAIS_E0_E4_2026-11-06.md`).

## 5. Limites

- Clôtures Yahoo, pas Deriv : les CFD Deriv ont des heures de cotation et des clôtures
  différentes ; les coûts sont ceux de Deriv aujourd'hui appliqués à tout l'historique.
- Yahoo n'a pas d'historique des actions radiées : le biais du survivant **surestime** les
  actions. La conclusion négative en est renforcée, pas affaiblie.
- Filtre de régime absent du balayage (trop long sur 124 marchés). Il améliore le forex de
  production ; rien ne dit qu'il changerait le signe des 31 marchés perdants, dont la perte
  vient des swaps.
- 124 essais : les 2 « gagnants » (NVDA, SOL) sont attendus par le hasard et ne doivent pas
  être retestés isolément, ce serait les choisir après coup.
- Le sens asymétrique des swaps (certains marchés sont gratuits à la vente) n'a pas été
  exploité : choisir le sens d'après le coût serait une stratégie différente, à tester comme
  telle, pas un ajout au bot.
