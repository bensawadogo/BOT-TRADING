# AGENTS.md

Les conventions du projet sont dans **CLAUDE.md** (architecture, règles, commandes) et
l'état des travaux dans **TASKS.md**. Les trois règles à ne jamais enfreindre :

1. Le live (`trend_bot/runner.py`) et le backtest (`trend_bot/backtest.py`) appellent les
   mêmes fonctions ; `test_live_egale_backtest` doit rester vert.
2. Une stratégie n'entre dans le bot que si elle bat le hasard après les coûts réels du
   courtier (`scripts/couts_mt5.py`), sur une période de test jamais utilisée pour la régler.
3. Aucun secret dans le code ; aucun compte réel sans `MT5_ALLOW_REAL=1`.
