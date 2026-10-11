# Bot de tendance MT5 : guide d'utilisation

## Ce que fait le bot
Une fois par jour (7 h UTC, en semaine) :
1. **Signal** : pour chaque marché, la moyenne des directions sur 1, 3 et 12 mois (de −1 à +1).
2. **Régime** : un modèle à sauts, entraîné sur 1 000 jours, coupe la position quand il contredit la tendance.
3. **Taille** : les positions sont dimensionnées pour une volatilité de **10 %/an** pour le portefeuille entier, avec un levier brut d'au plus 3.
4. **Sécurité** :
   - drawdown ≥ 10 % : exposition divisée par 2 ;
   - drawdown ≥ 20 % : **tout est fermé** jusqu'à ta remise à zéro (`--reset`).
5. **Ordres** : uniquement si la position doit changer de sens ou d'au moins 25 % (cela évite des frais inutiles).

Chaque décision est écrite dans `data/trend_journal.csv`, et dans Telegram si les variables `TELEGRAM_*` sont remplies.

## Installation (Windows, une seule fois)
1. Installe **MetaTrader 5** (Deriv MT5 ou ton courtier) et connecte-toi à un **compte démo**. Si possible, choisis un compte **sans swap** : les frais de financement de nuit pèsent lourd sur une stratégie qui garde ses positions des semaines.
2. Dans le dossier du projet :
   ```
   pip install -r requirements-trend.txt
   ```
3. Dans MT5, appuie sur Ctrl+M et vérifie les noms des marchés. Si un nom diffère (par exemple « US Tech 100 »), ajoute-le dans `trend_bot/config.py` (`UNIVERSE`).

## Utilisation
| Commande | Effet |
|---|---|
| `python main.py trend` | un cycle **en simulation** : calcule tout, n'envoie aucun ordre |
| `python main.py trend --live` | un cycle avec **vrais ordres** (refusé sur un compte réel) |
| `python main.py trend --live --loop` | tourne en continu, un cycle par jour à 7 h UTC, en semaine |
| `python main.py trend --reset` | remet à zéro le coupe-circuit après un drawdown de 20 % |
| `python main.py trend --paper data/fred` | même cycle sur des CSV, sans MT5 |

Un compte réel est refusé tant que `MT5_ALLOW_REAL=1` n'est pas dans `.env`. **Fais au moins 3 mois en démo d'abord.**

## Ce qu'on peut attendre
Résultats du backtest dans `docs/reports/TREND_BOT_BACKTEST_2026-10-08.md`. C'est une stratégie lente :
- des mois perdants, voire des années plates, sont normaux ;
- les gains viennent de quelques grandes tendances par an.

Ne pas la juger sur une semaine.

## À ne pas faire
- Ne trade pas à la main les marchés du bot sur le **même compte**. En compte netting, MT5 fusionne tout en une seule position par marché, et le bot ne s'y retrouverait plus.
- Ne lance pas deux `--loop` en même temps.
