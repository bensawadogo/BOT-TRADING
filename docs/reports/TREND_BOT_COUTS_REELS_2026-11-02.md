# Bot de tendance aux coûts réels de Deriv MT5 (02/11/2026)

*Coûts relevés sur le compte Deriv-Demo (`scripts/couts_mt5.py tendance`) puis rejoués
sur 1985-2026 avec la logique exacte du bot :
`python -m trend_bot.backtest data/fred data/couts_mt5.json`.*

## 1. Ce qui a été corrigé avant de mesurer
Le live ne tradait pas ce qui avait été backtesté :
- **Bougie du dimanche.** Deriv MT5 crée une bougie D1 le dimanche sur le forex (1 sur 6).
  Les horizons « 1/3/12 mois » du live faisaient donc 3,5/10,5/42 semaines, et la
  volatilité était sous-estimée d'environ 9 %. Désormais : calendrier lundi-vendredi
  partout (`weekdays_only`).
- **Cryptos 7 j/7.** En simulation, la volatilité de BTC était estimée à 51 % au lieu de
  60 %, et le portefeuille tournait à 12 % de volatilité au lieu de 10 %. Même correction.
- **Filtre de régime.** Le live ré-entraînait chaque jour, le backtest tous les 21 jours.
  Les deux ré-entraînent maintenant le 1er de chaque mois, sur les 1 000 jours précédents.
- **Panier.** Le live tradait 19 marchés, dont 11 jamais testés ; le pétrole et le gaz du
  backtest en étaient absents.
- **Spécification NGAS incohérente** (contrat de 10 000, valeur de tick d'une unité). Le
  bot aurait cru qu'un lot valait 3 $ au lieu de 30 875 $, et passé un ordre 10 000 fois
  trop gros. La plus grande des deux valeurs est désormais retenue.

Le test `test_live_egale_backtest` vérifie que le live et le backtest prennent les mêmes
positions, avec des bougies du dimanche et une crypto 7 j/7.

## 2. Coûts relevés (compte démo, ticks à 7 h UTC sur 5 jours)
| Marchés | Demi-spread | Surcoût des swaps (marge du courtier) |
|---|---|---|
| Forex majeurs | 0,10 à 0,28 bp | 1,3 à 2,2 %/an du notionnel |
| NZD, NOK, SEK, MXN | 0,6 à 1,0 bp | 0,2 à 2,9 %/an |
| Nasdaq (US Tech 100) | 0,12 bp | 2,1 %/an |
| Pétrole (WTI, Brent) | 0,7 bp | **61 à 75 %/an** (long comme court) |
| Gaz naturel | 8 bp | 37 %/an en moyenne ; **−224 %/an** à l'achat |
| BTC, ETH (hors panier) | 0,15 à 1,3 bp | 15 à 20 %/an |

Le spread est 5 à 10 fois plus faible que ce que le backtest d'octobre supposait. Les
swaps, eux, n'y étaient pas du tout. Le modèle de swap a été vérifié sur la position BTC
réelle du 11/10 : 32,35 $ facturés en 22 jours, pour 29 $ attendus.

## 3. Résultats 1985-2026
| Variante | Rendement/an | Vol | Sharpe | t | DD max | 2008-16 | 2017-26 |
|---|---|---|---|---|---|---|---|
| Bot, frais du rapport d'octobre | 4,5 % | 8,8 % | 0,54 | 3,5 | 21,6 % | 0,51 | 0,03 |
| 14 marchés, spreads réels, sans swap | 5,9 % | 8,7 % | 0,71 | 4,6 | 20,0 % | 0,65 | 0,15 |
| 14 marchés, spreads réels **+ swaps** | **−1,7 %** | 7,4 % | −0,20 | −1,3 | **70,1 %** | −0,09 | −0,75 |
| Sans énergies, spreads réels, sans swap | 5,1 % | 8,4 % | 0,64 | 4,1 | 19,0 % | 0,50 | 0,11 |
| Sans énergies, spreads réels **+ swaps** | 2,4 % | 8,4 % | 0,33 | 2,1 | 29,4 % | 0,34 | −0,16 |

Le filtre de régime ré-entraîné chaque mois donne le même résultat que l'ancien schéma
(bot : Sharpe 0,54 contre 0,55 en octobre). Les corrections n'ont pas embelli le backtest.

## 4. Conclusions
1. **Avec ses swaps actuels, ce compte rend le bot perdant** dès que le pétrole ou le gaz
   sont dans le panier. Ils sont retirés du panier par défaut (`config.ENERGIES`).
2. **Sans énergies, avec swaps** : 2,4 %/an, et négatif depuis 2017. C'est l'espérance
   réaliste sur ce type de compte, avant même la variance de la démo.
3. **Sur un compte sans swap**, le bot retrouve 5,1 %/an (Sharpe 0,64). Le type de compte
   compte plus que n'importe quel réglage de la stratégie.

## 5. Limites
- Spreads de **démo**, relevés un seul jour : le compte réel peut être plus cher.
- Les swaps changent avec les taux d'intérêt : relever `couts_mt5.py` chaque mois.
- Le backtest compare des prix au comptant (FRED) et ne retire que la marge du courtier ;
  le portage (différentiel de taux) n'est compté ni d'un côté ni de l'autre.
- Les prix du pétrole et du gaz chez Deriv sont des CFD, pas le prix au comptant de FRED.
