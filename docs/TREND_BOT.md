# Bot de tendance MT5 : guide d'utilisation

## Ce que fait le bot
Une fois par jour (7 h UTC, en semaine), sur la clôture de la veille :
1. **Signal** : pour chaque marché, la moyenne des directions sur 1, 3 et 12 mois (de −1 à +1).
2. **Régime** : un modèle à sauts, ré-entraîné le 1er jour de chaque mois sur les 1 000 jours
   précédents, coupe la position quand il contredit la tendance.
3. **Taille** : volatilité visée de **10 %/an** pour le portefeuille entier, levier brut ≤ 3.
4. **Sécurité** :
   - drawdown ≥ 10 % : exposition divisée par 2 ;
   - drawdown ≥ 20 % : **tout est fermé** jusqu'à ta remise à zéro (`--reset`).
5. **Ordres** : uniquement si la position doit changer de sens ou d'au moins 25 %.
   Les positions du bot sur un marché sorti du panier sont fermées.

Le live et le backtest appellent les mêmes fonctions sur les mêmes séries (bougies du lundi
au vendredi) ; un test le vérifie à chaque modification.

Garde-fous d'exécution : le bot refuse d'envoyer un ordre si « Algo Trading » est coupé dans
MT5 ; il ne passe d'ordre qu'entre 7 h et 20 h UTC ; et il **reporte de 30 min** toute nouvelle
exposition située à moins de 30 min d'une annonce économique à fort impact sur les devises du
marché (calendrier gratuit, `trend_bot/news.py`). Une réduction ou une fermeture n'est jamais
retardée, et la décision (poids) ne change pas : le backtest reste valable. `TREND_NEWS=0`
désactive le filtre.

Chaque décision est écrite dans `data/trend_journal.csv` avec sa **raison** en clair
(tendance, régime, volatilité, coupe-circuit, zone neutre) ; l'équité dans
`data/trend_equity.csv`, et un résumé part sur Telegram si `TELEGRAM_*` est rempli.

## Installation (Windows, une seule fois)
1. Installe **MetaTrader 5** (Deriv MT5) et connecte-toi à un **compte démo**.
   Dans MT5 : bouton **Algo Trading** activé (sinon les ordres sont refusés :
   `retcode=10027 AutoTrading disabled by client`).
2. Dans le dossier du projet :
   ```
   py -3.13 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   copy .env.example .env
   ```
3. Vérifie : `.venv\Scripts\python.exe main.py` (un cycle en simulation, aucun ordre).

## Utilisation
| Commande | Effet |
|---|---|
| `python main.py` | un cycle **en simulation** : calcule tout, n'envoie aucun ordre |
| `python main.py trend --live` | un cycle avec **vrais ordres** (refusé sur un compte réel) |
| `python main.py trend --live --loop` | tourne en continu, un cycle par jour à 7 h UTC, en semaine |
| `python main.py trend --reset` | remet à zéro le coupe-circuit après un drawdown de 20 % |
| `python main.py dashboard` | suivi : équité, dernier cycle, ordres refusés |

Un compte réel est refusé tant que `MT5_ALLOW_REAL=1` n'est pas dans `.env`.

## Faire tourner la démo 3 mois
`--loop` ne tourne que si le PC est allumé à 7 h UTC (MT5 est lancé automatiquement par le bot
s'il est fermé, avec le dernier compte connecté). En place depuis le 10/11/2026 :
- **Le bot bloque la veille automatique** tant qu'il tourne (`runner.empecher_la_veille`, même
  mécanisme qu'un lecteur vidéo). Il ne bloque ni une mise en veille manuelle ni un redémarrage.
- **Tâche planifiée « BOT-TRADING trend_bot »** (`scripts/lancer_bot.ps1`) : à l'ouverture de
  session et tous les jours à **06:45 UTC en réveillant le PC**, elle lance le bot s'il ne tourne
  pas déjà (jamais deux exemplaires). Après un redémarrage, la session doit s'ouvrir (connexion
  automatique Windows, ou se connecter avant 7 h).
- Ce que seul toi peux régler : les mises à jour Windows qui redémarrent la nuit, et l'autorisation
  des « minuteurs de réveil » dans les options d'alimentation (souvent coupés sur batterie).
- **VPS Windows** (quelques $/mois) reste la solution sans dépendance au PC.

Jalon : au moins **63 jours de cotation** dans `data/trend_equity.csv` avant de juger.

## Ce qu'on peut attendre
**Rien de régulier.** Aux coûts réels de Deriv, le panier est à zéro depuis 1996 (Sharpe +0,06,
+0,1 %/an, drawdown 30 %) et négatif depuis 2017 ; le Sharpe de 0,28 sur 1985-2026 vient entièrement
de 1985-1995 (`docs/reports/VERIFICATION_LOGIQUE_2026-11-10.md`). La démo valide la mécanique, pas un
revenu. Pour mémoire, les chiffres sur tout l'historique :

Backtest 1985-2026 aux coûts réels de Deriv MT5
(`docs/reports/TREND_BOT_COUTS_REELS_2026-11-02.md`), panier par défaut (forex + Nasdaq) :
- **avec swaps** (compte standard) : ≈ 2,4 %/an, drawdown max 29 %, négatif sur 2017-2026 ;
- **sans swap** : ≈ 2,8 %/an, drawdown max 26 %, une fois les frais d'administration de
  Deriv comptés (`docs/reports/COMPTE_SANS_SWAP_2026-11-06.md` ; le chiffre de 5,1 % du
  02/11 les supposait nuls).

Sur 10 000 $, cela fait de l'ordre de 240 à 280 $ par an en moyenne, avec des années
perdantes. C'est une stratégie lente : on la juge sur des années, pas sur des semaines.
Le pétrole et le gaz sont exclus du panier : avec leurs swaps (≈ 60-75 %/an), le bot perd
de l'argent.

## Capital minimum
Le bot ne peut pas trader moins que le lot minimum du courtier. Quand un marché tombe
sous ce minimum, le journal l'indique (`sous le lot minimum (équité nécessaire ≈ …)`) et
le résumé du cycle liste ces marchés. Ordres de grandeur sur Deriv MT5 : un lot minimum
vaut ~560 à 1 340 $ sur le forex et ~2 800 $ sur le Nasdaq, pour des poids de 10 à 60 %
de l'équité. En dessous d'environ 2 000 $, le bot ne passe presque aucun ordre. Avec
~10 000 $, le forex est tradé, mais pas toujours le Nasdaq.

## À ne pas faire
- Ne trade pas à la main les marchés du bot sur le **même compte** (en netting, MT5 fusionne
  les positions).
- Ne lance pas deux `--loop` en même temps.
- N'ajoute pas un marché dans `TREND_SYMBOLS` sans l'avoir backtesté avec ses coûts réels.
