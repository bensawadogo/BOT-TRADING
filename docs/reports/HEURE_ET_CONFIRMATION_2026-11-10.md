# Deux essais demandés : heure d'exécution et délai de confirmation (10/11/2026)

*Deux questions posées après le premier cycle réel : « le bot devrait trader la session
américaine (13 h UTC, plus de liquidité) » et « le bot devrait apprendre des faux mouvements ».
Scripts : `scripts/couts_mt5.py` (fonction `spread_relatif`) et `lab/confirmation.py`
(recherche, rien n'est branché au bot). Journal : `logs/confirmation.log`.*

## 1. Spread par heure : Deriv a des spreads quasi fixes

Demi-spread médian sur les ticks de 10 jours ouvrés (31/10 → 07/11/2026), en points de base,
par fenêtre d'une heure UTC :

| | 7 h | 9 h | 13 h | 14 h | 15 h | 16 h | 20 h |
|---|---|---|---|---|---|---|---|
| EURUSD | 0,18 | 0,20 | 0,22 | 0,22 | 0,22 | 0,22 | 0,18 |
| GBPUSD | 0,22 | 0,20 | 0,22 | 0,22 | 0,22 | 0,20 | 0,22 |
| USDJPY | 0,10 | 0,09 | 0,09 | 0,09 | 0,09 | 0,09 | 0,10 |
| AUDUSD | 0,18 | 0,18 | 0,15 | 0,15 | 0,15 | 0,15 | 0,15 |
| USDCAD | 0,12 | 0,12 | 0,11 | 0,11 | 0,11 | 0,11 | 0,11 |
| USDCHF | 0,18 | 0,18 | 0,18 | 0,18 | 0,18 | 0,18 | 0,18 |
| NZDUSD | 0,54 | 0,53 | 0,53 | 0,53 | 0,53 | 0,53 | 0,53 |
| USDMXN | 1,01 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 | 1,00 |
| USDNOK | 1,04 | 0,99 | 1,08 | 0,98 | 1,04 | 1,09 | **1,61** |
| USDSEK | 0,79 | 0,76 | 0,74 | 0,74 | 0,78 | 0,82 | **1,77** |
| NAS100 | 0,12 | 0,12 | 0,12 | 0,12 | 0,12 | 0,12 | 0,12 |
| **Médiane** | 0,18 | 0,20 | 0,22 | 0,22 | 0,22 | 0,20 | 0,18 |

- Le signal du bot vient de bougies journalières terminées : l'heure d'exécution ne change
  que le prix d'entrée, jamais la décision. Les écarts entre 7 h et 13-16 h sont de l'ordre du
  centième de point de base, sur un coût payé une fois par changement de position tenue des
  semaines : **sans effet mesurable**. Seule 20 h est à éviter (NOK et SEK doublent).
- **Le cycle reste à 7 h UTC** (`TREND_RUN_HOUR_UTC`). Trader les mouvements de la session
  elle-même est une autre stratégie (scalping), mesurée négative le 02/11.

## 2. Délai de confirmation : « attendre que le mouvement soit vrai » coûte du rendement

Backtest de production (panier de 11 marchés, régime, coûts Deriv : demi-spread + surcoût de
swap), 1985-2026, seul le signal est remplacé : la tendance n'est suivie qu'après N clôtures
consécutives du même signe (sinon on garde la dernière valeur confirmée). Causal.

| Variante | Sharpe | CAGR | DD | 1996-26 | 2008-26 | P > T |
|---|---|---|---|---|---|---|
| **T** tendance (production) | **+0,28** | +2,0 % | 30 % | +0,06 | +0,04 | |
| C3, confirmé 3 jours | +0,14 | +0,8 % | 40 % | −0,06 | −0,07 | 0,03 |
| C5, confirmé 5 jours | +0,13 | +0,8 % | 34 % | −0,02 | +0,01 | 0,07 |
| C10, confirmé 10 jours | +0,09 | +0,4 % | 41 % | −0,02 | −0,07 | 0,04 |
| Sharpe dégonflé du meilleur (C3, 3 essais) | 0,77 (seuil 0,95) | | | | | |

- **Toutes les variantes font moins bien que le bot**, sur toutes les périodes, et le
  drawdown augmente. P(C > T) = 0,03 à 0,07 : le bot d'origine bat ses variantes dans 93 à
  97 % des rééchantillonnages. Attendre la confirmation fait rater le début des mouvements,
  là où la tendance gagne, sans éviter assez de faux départs pour compenser.
- Un « faux mouvement » en suivi de tendance n'est pas une erreur à apprendre : c'est le
  coût normal d'une stratégie qui perd sur 55 à 65 % de ses positions et gagne sur les
  autres. Les variantes « éviter les faux mouvements » déjà mesurées (tendance continue,
  désaccord lent/rapide, régime adouci : `ESSAIS_E0_E4_2026-11-06.md`) donnaient la même réponse.

## 3. Ce que ça change

Rien dans le bot. Les deux idées sont **classées** : pas d'heure à changer, pas de délai de
confirmation à ajouter. Ne pas les retester avec d'autres réglages.

## 4. Limites

- Spreads relevés sur 10 jours de novembre 2026 ; Deriv peut les changer.
- Délais testés : 3, 5, 10 jours, choisis avant de voir le résultat ; trois essais comptés
  dans le Sharpe dégonflé.
