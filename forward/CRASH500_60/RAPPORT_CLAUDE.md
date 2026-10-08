# Rapport forward-demo - mesure de l'edge en conditions reelles

- **Symbole** : `CRASH500` - granularite 60s
- **Duree**   : 270.5 min - 1696 decisions, 1899 issues revelees
- **Spread aller simple** : 0.2000%
- **Base UP (baseline drift)** : 90.0 % (190 UP / 21 DOWN)

> Seuil de reference : la baseline UP (drift brut). Un modele
> dont le WR = baseline copie le drift -> **aucun edge**.

## Resultats par-modele (forward reel, issues horodatées)

| Modele | Decisions | WR | Edge | p-value | PnL net | Verdict |
|---|---|---|---|---|---|---|
| BaselineUP | 211 | 90.0 % | - | - | 0.0000 | -0 | - drift |
| Ensemble4sur5 | 177 | 87.0 % | 90.0 % | -3.0 pp | 0.0000 | -0 | VERT_EDGE |
| HMM | 211 | 80.1 % | 90.0 % | -10.0 pp | 0.0000 | -0 | VERT_EDGE |
| Kalman | 188 | 80.8 % | 90.0 % | -9.2 pp | 0.0000 | -0 | VERT_EDGE |
| Momentum | 176 | 80.7 % | 90.0 % | -9.4 pp | 0.0000 | -0 | VERT_EDGE |
| Online | 181 | 82.3 % | 90.0 % | -7.7 pp | 0.0000 | -0 | VERT_EDGE |
| RSI | 193 | 80.8 % | 90.0 % | -9.2 pp | 0.0000 | -0 | VERT_EDGE |
| TrendStrength | 146 | 87.7 % | 90.0 % | -2.4 pp | 0.0000 | -0 | VERT_EDGE |
| XGBoost | 211 | 73.0 % | 90.0 % | -17.1 pp | 0.0000 | -0 | VERT_EDGE |

## Verdict global

AUCUN depasse la baseline UP.
Sur 1899 issues, TOUS les modeles (Ensemble4sur5, HMM, Kalman, Momentum, Online, RSI, TrendStrength, XGBoost) ont :
- WR < baseline UP (90.0%)
- PnL net negatif
- Aucun edge exploitable (WR < baseline = copie de la derive)

### A remettre a Claude
1. Revoir la logique de verdict (VERDICT VERT_EDGE erronee)
2. Tester sur d'autres indices pour verifier si l'absence de edge est specifique a CRASH500 M1
3. Revoir la configuration (tick_interval, contracts, spread) pour trouver des conditions ou un edge est possible
4. Tester avec triple-barriere symetrique comme target (anti-drift)
5. Confronter WR live vs permutation_test_score sur ce flow
