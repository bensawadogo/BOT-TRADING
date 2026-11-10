# Scalping sur Deriv MT5 : analyse sur données réelles (02/11/2026)

*Compte Deriv-Demo. Spread mesuré sur les ticks (ask − bid) à 8 h, 13 h et 16 h UTC sur
5 jours ; mouvements mesurés sur ~20 000 bougies M1 par marché. Reproductible :
`python scripts/couts_mt5.py scalping` (lecture seule, aucun ordre).*

## 1. Le spread face au mouvement

« Coût/mvt » = spread ÷ mouvement médian sur la durée de détention. « Réussite » = part
de trades gagnants nécessaire pour ne rien perdre avec un objectif et un stop placés à un
mouvement médian : 0,5 + spread / (2 × mouvement).

| Marché | Spread | 1 min | 5 min | 15 min | 60 min |
|---|---|---|---|---|---|
| EURUSD | 0,27 bp | 60 % → **80 %** | 25 % → 63 % | 14 % → 57 % | 7 % → 54 % |
| GBPUSD | 0,23 bp | 50 % → 75 % | 23 % → 62 % | 13 % → 56 % | 6 % → 53 % |
| USDJPY | 0,19 bp | 38 % → 69 % | 16 % → 58 % | 9 % → 55 % | 4 % → 52 % |
| Or (XAUUSD) | 0,37 bp | 24 % → 62 % | 10 % → 55 % | 6 % → 53 % | 3 % → 51 % |
| Nasdaq (US Tech 100) | 0,24 bp | 15 % → 57 % | 6 % → 53 % | 3 % → 52 % | 2 % → 51 % |
| S&P 500 | 0,40 bp | 42 % → 71 % | 18 % → 59 % | 10 % → 55 % | 5 % → 53 % |
| BTCUSD | 0,30 bp | 14 % → 57 % | 6 % → 53 % | 4 % → 52 % | 2 % → 51 % |
| Volatility 75 | 3,07 bp | 44 % → 72 % | 20 % → 60 % | 12 % → 56 % | 6 % → 53 % |
| Volatility 10 | 0,38 bp | 41 % → 71 % | 18 % → 59 % | 10 % → 55 % | 5 % → 53 % |
| Boom 500 / Crash 500 | 0,14 bp | 12 % → 56 % | 2 % → 51 % | 1 % → 51 % | 1 % → 50 % |

Plus la position est courte, plus le spread mange le mouvement. Sur EURUSD à 1 minute,
il faut gagner 8 trades sur 10 pour seulement ne rien perdre.

## 2. Y a-t-il quelque chose à prédire à 1-5 minutes ?

Scalpeur naïf : à chaque clôture M1, suivre (ou contrer) la minute écoulée, garder 1 ou
5 minutes, payer le spread. Sur les 11 marchés, l'avantage **brut** va de −0,13 à
+0,13 bp par trade, soit toujours moins que le spread. Résultat **net** : négatif partout,
de −0,10 à −3,2 bp par trade, avec t de −1,5 à −43. Les minutes successives sont
indépendantes : il n'y a pas de motif simple à exploiter.

Les indices synthétiques (Volatility, Boom/Crash, Step, Jump) sont produits par un
générateur aléatoire : par construction, leur prochaine minute est imprévisible.

## 3. Ce que cette mesure ne compte pas (et qui aggrave)
- **Spreads de démo** : le compte réel peut être plus cher, et les spreads s'élargissent
  aux annonces économiques et à l'heure du rollover (22-23 h UTC).
- **Glissement** : entre le signal et l'exécution, le prix bouge (latence internet).
  À 1 minute, quelques dixièmes de bp suffisent à doubler le coût.
- **Commissions** éventuelles selon le type de compte.

## 4. Ce que le projet a déjà mesuré
- 5 algorithmes en M5 et H1 (or, 8 paires forex, Volatility 75) : tous négatifs après
  coûts, milliers de trades (`BACKTEST_ALGOS_TENDANCE_2026-10-08.md`).
- BOOM/CRASH en M1 : aucun edge, le contrat testé n'existait pas
  (`AUDIT_BOOM_CRASH_2026-10-08.md`).
- Ensemble de 7 modèles en M1 à H4 : aucun modèle au-dessus du taux de base (archivé).

## 5. Verdict
**Ne pas scalper.** Le coût par trade dépasse l'avantage mesurable sur tous les marchés
testés, et l'écart se creuse à mesure que l'horizon raccourcit. Les études sur les
particuliers vont dans le même sens : parmi les 1 551 day traders brésiliens qui ont
persévéré plus de 300 séances, 17 seulement (1,1 %) gagnaient plus que le salaire minimum
(Chague, De-Losso et Giovannetti, *Day Trading for a Living?*, 2020). En Europe, l'ESMA a
relevé que 74 à 89 % des comptes de particuliers perdent de l'argent sur les CFD.

Pour reprendre l'idée un jour, une stratégie de scalping devrait, avant tout argent réel :
1. être écrite en entier AVANT le test (aucun réglage sur les données de test) ;
2. battre des entrées au hasard avec les mêmes sorties (`lab/engine.random_baseline`) ;
3. rester positive avec le spread réel ×2 et 0,1 bp de glissement par trade ;
4. tenir sur au moins 1 000 trades hors échantillon, sur plusieurs marchés.
