# Phase 2 - Correctifs sécurité

Résumé:
Phase 2 + 2.5 complètes. Sécurité RCE Pickle + migration configuration + étiquetage données synthétiques.
294/294 tests passent.

CORRECTIONS:

1. Pickle RCE — SafeUnpickler (deriv/constants.py)
   Whitelist de modules types safe (numpy, pandas, sklearn, etc.)
   builtins whitelisté uniquement types sûrs (float, int, str, dict, list...)
   eval/exec/compile/open STRICTEMENT BLOQUÉS
   safe_pickle_load() dans 5 fichiers pickle:
   - ensemble_predictor.py (3x: build_features, session_filter, models_dict)
   - online_learner.py (1x: OnlineLearner._load)
   - hmm_regime.py (1x: charge modèle)

2. Token Telegram — Migration vers TelegramNotifier (deriv/telegram.py)
   - Nouveau module deriv/telegram.py: classe TelegramNotifier
   - bot_executor.py: self.telegram = TelegramNotifier(...)
   - logger.warning(): token masqué dans messages d'erreur
   - URL API contient token mais jamais loguée

3. SafeUnpickler bypass evaluation (constants.py)
   - builtins.float/int/str/list/dict/... autorisés (nécessaires pour pickle)
   - builtins.eval/exec/compile/open/breakpoint/... BLOQUÉS
   - Test 5/5: safe data, os.system, subprocess, eval, sklearn models

4. getenv → Config migration
   - bot_executor.py: STAKE_AUTO → Config.AUTO_STAKE
   - trading_journal.py: DB_PATH → Config.JOURNAL_DB
   - train_and_backtest.py: DERIV_SYMBOL → Config.SYMBOL
   - Config enrichi: AUTO_STAKE + JOURNAL_DB ajoutés
   - 0 os.getenv restant hors Config

5. BOM removal
   - deriv/train_and_backtest.py: BOM UTF-8 retiré

6. Dashboard — Données synthétiques étiquetées (dashboard/app.py)
   - Ajout bannière MODE SYNTHÉTIQUE/OFFLINE quand données non live
   - Statut WebSocket dynamique (Connecté/Offline selon is_live)
   - Solde/P&L dynamiques (capital_input) au lieu de 10,000$ statique
   - Crypto section: tous les gains/régimes étiquetés SYNTH/SIM/DEMO
   - Avant: données inventées présentées comme réelles
   - Après: toutes les données clairement étiquetées synthétiques

VERIFICATION:
- Safe unpickler: 5/5 tests passent (eval/exec bloqués, float/int OK)
- Syntaxe: tous les fichiers modifiés passent py_compile
- Tests: 294/294 OK
- BOM-free UTF-8: vérifié
- state.json: os.getenv remaining = 0

PHASE 3 reporté: REPO_TECH_DEBT.md (280+ problèmes), forward_demo PnL, 280+ problèmes tech debt
