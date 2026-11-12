# Bot de tendance MT5 (`trend_bot/`) : backtest de la logique exacte

*8 octobre 2026. `python -m trend_bot.backtest data/fred` rejoue jour par jour les fonctions mêmes
du runner live : signaux, régime, poids, coupe-circuit, zone neutre.*

## Pourquoi MT5 et pas l'API Deriv actuelle
Vérifié sur l'API publique Deriv le 08/10/2026 :
- **Multiplicateurs** forex, or et BTC : ×100 minimum, donc la mise est perdue dès 1 % de mouvement contraire. Incompatible avec des positions tenues plusieurs semaines.
- **Indices boursiers** : options binaires uniquement.
- **Historique journalier** : les 260 derniers jours seulement.

Deriv MT5 (CFD avec marge) n'a pas ces limites et fournit de longs historiques.

## Résultats : 14 marchés, 1985-2026 (10 paires forex, pétrole WTI et Brent, Nasdaq, gaz)
| Variante | CAGR | Vol | Sharpe | t | DD max | Rotation/an | 1990-07 | 2008-16 | 2017-26 |
|---|---|---|---|---|---|---|---|---|---|
| A Tendance seule | 4,2 % | 10,5 % | 0,45 | 2,88 | 41,9 % | 112 | 0,62 | 0,30 | −0,19 |
| B + filtre de régime | 4,9 % | 10,6 % | 0,51 | 3,28 | 28,9 % | 106 | 0,56 | 0,50 | −0,00 |
| **C + coupe-circuit = le bot** | 4,6 % | 8,8 % | **0,55** | **3,56** | **24,0 %** | 77 | 0,54 | 0,55 | 0,03 |
| D bot sans zone neutre | 4,2 % | 8,9 % | 0,51 | 3,27 | 24,9 % | 96 | 0,53 | 0,44 | 0,01 |

Chaque couche du bot améliore le Sharpe et réduit le drawdown.

Années du bot :
- 59 % des années sont positives ;
- pire année : −11,4 % (2019) ;
- meilleure année : +41,6 % (2008).

## Les limites, à connaître avant de mettre de l'argent
1. **Sensibilité aux frais.** Avec des frais de transaction ×3, le Sharpe passe de 0,55 à 0,14 ; avec ×5, il tombe à −0,17.
2. **La période récente est faible.** Sur 2008-2026, le Sharpe est de 0,28 avec les frais de base (t ≈ 1,2, non significatif) et de −0,10 avec des frais ×3.
3. **Réduire la rotation n'a pas aidé.** Protocole fixé avant le test : choix sur 1985-2007, vérification sur 2008-2026.

   | Variante | Rotation/an | Sharpe 1985-07 (×3 frais) | Sharpe 2008-26 (frais ×1) |
   |---|---|---|---|
   | Quotidien, zone neutre 25 % (défaut) | 77 | 0,34 | 0,28 |
   | V1 hebdomadaire (choisi sur 1985-2007) | 36 | 0,58 | 0,16 |
   | V2 zone neutre 50 % | 63 | 0,37 | 0,20 |
   | V3 hebdomadaire + zone 50 % | 30 | 0,56 | −0,01 |

   V1 a été retenu sur la première période mais fait moins bien ensuite : le réglage par défaut est conservé.
4. **Le swap (financement de nuit des CFD) n'est pas inclus.** Il peut coûter plusieurs % par an du notionnel. Deriv propose des comptes MT5 **sans swap** sur certains actifs : à privilégier. Vérifier les colonnes « swap » dans MT5 (clic droit sur le marché → Spécification).
5. **Le panier testé n'est pas celui du live.** Le backtest porte surtout sur le forex (données FRED), alors que le bot live trade aussi des indices, l'or et des cryptos. Ces actifs ont historiquement de meilleures tendances que le forex, mais nous n'avons pas pu le vérifier ici.

## Verdict
C'est la seule stratégie de tout le projet dont l'avantage est **documenté** (sur plus d'un siècle chez AQR) **et mesuré** ici (t = 3,6 sur 40 ans). L'avantage récent est faible et dépend fortement des frais.

Plan :
1. **3 mois minimum en compte démo** (`python main.py trend --live --loop` sur un compte démo), en suivant `data/trend_journal.csv` ;
2. un compte sans swap ;
3. au départ, seulement les marchés à faible spread.
