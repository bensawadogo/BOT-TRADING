# CLAUDE.md — Guide du projet

Point d'entrée pour Claude et tout contributeur : ce que fait le projet, où sont les
choses, les règles à respecter, et l'état actuel.

## 1. Le projet en bref

Un **bot de tendance multi-marchés** (`trend_bot/`) pour MetaTrader 5 (Deriv MT5 ou autre
courtier), et un **labo de recherche** (`lab/`) qui teste les idées avant qu'elles
n'arrivent dans le bot.

- La stratégie : tendance 1/3/12 mois, volatilité cible du portefeuille 10 %/an, filtre de
  régime (modèle à sauts), coupe-circuit de drawdown. Une décision par jour.
- Statut : **démo uniquement** (`MT5_ALLOW_REAL=0`), pour valider la mécanique, pas un revenu.
  **Aux coûts réels de Deriv MT5, le bot est à zéro depuis 1996 (Sharpe +0,06, +0,1 %/an) et
  négatif depuis 2017 ; tout l'avantage du backtest 1985-2026 (Sharpe 0,28) vient de
  1985-1995** (`docs/reports/VERIFICATION_LOGIQUE_2026-11-10.md`). Le filtre de régime réduit le
  drawdown (42 % → 30 %) sans créer de rendement. Le compte sans swap n'est pas gratuit
  (frais d'administration) : 2,8 %/an (`docs/reports/COMPTE_SANS_SWAP_2026-11-06.md`). Le signal de
  portage ne passe pas chez Deriv ; sur futures sans swap il ferait 0,63 de Sharpe, décorrélé de la
  tendance : seule piste au-dessus du seuil, sur un support non détenu (`docs/reports/PORTAGE_2026-11-10.md`). Tendance continue, désaccord
  lent/rapide, régime adouci, Fibonacci : aucune variante ne bat le bot d'origine
  (`docs/reports/ESSAIS_E0_E4_2026-11-06.md`). Le bot n'a aucun stop par position (seul le coupe-circuit du
  portefeuille protège) : le stop serré gagne 0,03 de Sharpe sans être significatif et ne réduit pas les profits
  rendus, les stops larges ne se déclenchent presque jamais (`docs/reports/STOPS_2026-11-12.md`). **124 autres marchés Deriv** (crypto, actions,
  indices, métaux, ETF, agricoles, croisés) balayés le 10/11/2026 : aucun avantage de tendance,
  31 perdants significatifs à cause des swaps ; la crypto gagne par bêta et fait moins bien
  qu'acheter-et-garder (`docs/reports/SCAN_MARCHES_2026-11-10.md`). Le panier reste le forex + Nasdaq.
- L'ancien bot Deriv (ensemble de 7 modèles, drift BOOM/CRASH, options binaires) n'avait
  **aucun edge** mesuré : il est archivé sous le tag git `archive/deriv-ensemble`
  (`git checkout archive/deriv-ensemble` pour le consulter). Ne pas le ressusciter.

## 2. Arborescence

```text
BOT-TRADING/                  (C:\Users\sawad\BOT-TRADING)
├── main.py                   # python main.py [trend|dashboard] [--live] [--loop] [--reset] [--paper DIR]
├── trend_bot/                # le bot
│   ├── signals.py            # alpha : tendance, volatilité ex-ante, calendrier lun-ven
│   ├── risk.py               # filtre de régime (ré-entraînement mensuel), coupe-circuit
│   ├── portfolio.py          # poids (vol cible), lots, zone neutre
│   ├── runner.py             # un cycle live : données → décisions → ordres → journal
│   ├── backtest.py           # rejoue EXACTEMENT les mêmes fonctions sur l'historique
│   ├── data.py               # séries FRED du backtest + correspondance avec les marchés
│   ├── config.py             # panier, paramètres (.env), chemins
│   └── brokers/              # mt5.py (MetaTrader5), paper.py (CSV, tests)
├── lab/                      # recherche : moteur causal, HMM causal, Fibonacci, carry.py (portage), confirmation.py, stops.py
├── dashboard/app.py          # Streamlit : lit le journal du bot, rien d'inventé
├── scripts/                  # fetch_fred.py (données), couts_mt5.py (coûts réels), scan_marches.py (autres marchés),
│                             # lancer_bot.ps1 (tâche planifiée : réveil 06:45 UTC, un seul exemplaire)
├── docs/                     # TREND_BOT.md (guide), reports/ (audits et backtests datés)
├── tests/                    # main.py, dashboard
└── data/                     # local, ignoré par git : état, journal, équité, FRED
```

## 3. Règles non négociables

1. **Le live trade ce qui a été backtesté.** `runner.py` et `backtest.py` appellent les
   mêmes fonctions sur les mêmes séries. Le test `test_live_egale_backtest` le vérifie :
   toute modification de la logique doit le garder vert. Ne jamais dupliquer une règle.
2. **Calendrier lundi-vendredi** (`weekdays_only`). Deriv MT5 crée une bougie D1 le
   dimanche sur le forex et les cryptos cotent 7 j/7 ; les horizons en bougies et
   ANN = 252 supposent 5 jours par semaine.
3. **Le panier par défaut = les marchés backtestés et tradables** (`config.UNIVERSE`).
   Les énergies (swaps prohibitifs) et les marchés jamais testés ne sont tradés que sur
   demande explicite (`TREND_SYMBOLS`).
4. **Causalité** : une décision à la clôture t n'utilise que des données ≤ t. Tout
   nouveau signal doit avoir un test « le futur ne change pas le passé ».
5. **Une nouvelle stratégie doit battre le hasard après les coûts réels du courtier**
   (spread relevé sur les ticks + swaps : `scripts/couts_mt5.py`), sur une période de
   test jamais utilisée pour la régler. Sinon, elle ne va pas dans le bot.
6. **Annonces et exécution** : le filtre d'annonces (`trend_bot/news.py`) ne fait que REPORTER
   l'ouverture d'un ordre, jamais une réduction, et ne change pas les poids (live = backtest).
   **Sécurité** : jamais de secret dans le code (`.env`, ignoré par git) ; compte réel
   refusé sans `MT5_ALLOW_REAL=1` ; le bot ne touche qu'à SES positions (numéro magique).
7. **Racine propre** : logs dans `logs/`, scripts dans `scripts/`, rapports datés dans
   `docs/reports/`. Pas de fichier temporaire à la racine.

## 4. Commandes

Prérequis : Python 3.12 ou 3.13 (le Python 3.14 système de cette machine plante dans
`hmmlearn`), terminal MT5 installé et connecté pour le live.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q                       # tests
.venv\Scripts\python.exe main.py                            # un cycle en simulation
.venv\Scripts\python.exe main.py trend --live --loop        # démo : un cycle par jour
.venv\Scripts\python.exe main.py dashboard                  # suivi
.venv\Scripts\python.exe scripts\fetch_fred.py data\fred    # données du backtest
.venv\Scripts\python.exe scripts\couts_mt5.py tendance      # coûts réels → data\couts_mt5.json
.venv\Scripts\python.exe -m trend_bot.backtest data\fred data\couts_mt5.json
.venv\Scripts\python.exe -m trend_bot.backtest data\fred data\couts_mt5.json --essais   # 5 essais, ≈ 50 min
.venv\Scripts\python.exe scripts\scan_marches.py fetch|couts|run|classes   # balayage d'autres marchés (≈ 1 h 30)
```

## 5. Où en est le projet

Voir `TASKS.md`. Les conclusions chiffrées sont dans `docs/reports/` (une conclusion =
un rapport daté, jamais réécrit après coup).
