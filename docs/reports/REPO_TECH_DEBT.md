# Dette technique & code spaghetti — Synthèse
## C:\BOT-TRADING — 16 septembre 2026

> 4 agents analytiques en parallèle ont produit 280+ problèmes identifiés.
> Ce document les priorise et les traduit en plan d'action concret.

---

## TABLEAU DE BORD — SÉVÉRITÉ

| Sévérité | Count | Impact principal |
|----------|-------|------------------|
| CRITIQUE | 14 | Failles de fiabilité, pertes financières, données corrompues |
| MAJEUR | 55 | Maintenance risquée, tests fragiles, divergence de résultats |
| MINEUR | 80+ | Confusion, dette cosmétique, dette de documentation |

---

## CRITIQUE — À corriger URGEMMENT

### 1. 🔴 Backtest entraîne sur les données de test + écrase les modèles de production
- `deriv/train_and_backtest.py:84-88, 118-124` + `deriv/ensemble_predictor.py:619-626`
- **Impact** : les modèles sont entraînés sur triple-barrière (features/labels/train) mais le backtest mesure `close[t+1] > close[t]`. Le WR publié (70-90%) ne mesure PAS la compétence du modèle entraîné → conclusions fausses depuis des mois.
- **Plus** : `train_and_backtest.py` peut écraser les pickles de production (hmm_deriv.pkl, xgb_deriv.pkl).
- **Action** : utiliser un contrat de target unique, entraîner dans un répertoire temporaire, désactiver les sauvegardes pendant le backtest.

### 2. 🔴 Dashboard affiche des données factices comme réelles
- `dashboard/app.py:157-167, 201-296`
- **Impact** : le dashboard affiche connexion, solde, P&L, trades et statut Freqtrade provenant de **zéro source réelle**. Un utilisateur prend des décisions financières sur des valeurs inventées.
- **Action** : état de données explicite (live/offline/unknown/synthetic), masquer les métriques non disponibles, alerte si API Deriv en panne.

### 3. 🔴 Échec API silencieux dans le dashboard
- `dashboard/app.py:136-151` — `except Exception: pass`
- **Impact** : panne Deriv = mode simulation silencieux, aucune alerte fiable.

### 4. 🔴 Test max_stake_pollue os.environ sans cleanup
- `tests/test_deriv.py:48-52`
- **Impact** : `DERIV_AUTO_MAX_STAKE=5` persiste pour TOUS les tests suivants → résultats non-déterministes selon l'ordre d'exécution.

### 5. 🔴 Tests "unitaires" font des appels réseau réels
- `tests/test_deriv.py:20-40` — `test_connect_and_ping`, `test_get_ticks_live`
- **Impact** : `pytest tests/` sans réseau = échec bloquant. CI cassé.

### 6. 🔴 Scripts _*.py dans tests/ écrasent les modèles de production
- `tests/_retrain_xgb.py:78-82`, et tous les scripts `_*.py`
- **Impact** : `python tests/_retrain_xgb.py` → écrit dans `deriv/models/xgb_deriv.pkl` → modèle de production corrompu.

### 7. 🔴 Fixture conftest.py ne protège PAS tous les chemins de modèles
- `conftest.py:17-33` — patche `MODELS_DIR` de `ensemble_predictor` et `hmm_regime` mais OUBLIE `online_learner.MODELS_DIR`, `STATE_FILE`, `weekly_retrain.DEFAULT_STATE_FILE`.
- **Impact** : `OnlineLearner.__init__()` dans certains tests écrit dans `deriv/models/online_learner.pkl` → pollution production.

### 8. 🔴 Deux définitions de BREAKEVEN contradictoires dans le code
- `walk_forward_optimizer.py:46` : `BREAKEVEN = 0.556` (float)
- `risk_manager.py:43` : `BREAKEVEN_WINRATE = 55.6` (pourcentage)
- `symbol_optimizer.py:38` : `BREAKEVEN = 0.556` (copie)
- `journal_learner.py:151` : `wr_global >= 0.556` (copie)
- **Impact** : si le seuil change, 4 fichiers à modifier, risque d'incohérence.

### 9. 🔴 Target d'entraînement ≠ Target de backtest
- `deriv/train_and_backtest.py` vs `deriv/ensemble_predictor.py`
- **Impact** : même problème que le #1 — les WR publiés ne mesurent pas ce qui est entraîné.

### 10. 🔴 Le mode `all` de main.py bloque le dashboard
- `main.py:38-41` — Deriv démarre puis attend sa fin avant le dashboard.
- **Impact** : connexion Deriv bloquante = dashboard inaccessible.

### 11. 🔴 Online learner: vote online NE participe PAS à la règle 4/5
- `deriv/online_learner.py:23` — docstring claire, mais le `_ensemble_et_online()` dans `forward_demo.py:180-198` inclut Online dans les votes alors que le système officiel l'exclut.
- **Impact** : incohérence entre le forward demo et le bot réel.

### 12. 🔴 TradingJournal export_for_ml: 0 tests
- Module critique pour le ML, aucune couverture de test.

### 13. 🔴 DerivBot (deriv/bot.py): 0 tests
- L'orchestrateur principal du bot n'est testé par rien.

### 14. 🔴 Forward demo CRASH500: tous les modèles sous baseline UP
- Résultat actuel (657 issues, ~72 min écoulées) : WR 76-88%, edge -1.9 à -32pp, PnL négatif pour tous.
- **Impact** : confirme que le WR "89%" historique était un artefact de drift, pas de la compétence. HOLD confirmé.

---

## MAJEUR — À corriger dans la prochaine sprint

### Duplication de code entre modules
1. **Logique vote→proba** : `ensemble_predictor.py:368` (`_vote_from_proba`), `walk_forward_optimizer.py:473` (`_vote_to_proba`), `walk_forward_optimizer.py:261` (`_StatelessWrapper`), `ensemble_predictor.py:147` (`_EnsembleAdapter`) — 4 implémentations de la même règle avec des seuils légèrement différents.
2. **Calibration de seuil** : `ensemble_predictor.py:552` (`calibrate_global_threshold`) et `walk_forward_optimizer.py:488` (`_pooled_threshold`) — même algorithme.
3. **EnsembleAdapter vs StatelessWrapper** : 2 classes quasi-identiques pour le même but.
4. **log_ret** calculé dans `build_features()` (ligne 110) et `_hmm_features()` (ligne 361).

### God Objects
1. `EnsemblePredictor` : 1283 lignes, 7 sous-modèles + vote + calibration + save/load.
2. `WalkForwardOptimizer` : 1166 lignes, orchestre WFO + calibration + rapports + CLI.
3. `DerivDataCollector` : 335 lignes, 5 responsabilités (connexion, ticks, bougies, historique, pagination).
4. `XGBoostPredictor` : 253 lignes, train + predict + calibrate + save/load.

### Tests fragiles
1. **AST introspection** (`deriv/tests/test_anti_leakage.py:78-128`) : tests qui analysent le code source plutôt que le comportement → cassent à chaque refactoring.
2. **`__new__` bypass** (`tests/test_ensemble_predictor.py:170`) : contourne le constructeur, couplage total à l'implémentation interne.
3. **`generate_synthetic_ohlcv` dupliqué 4×** avec des RNG différents (seed global vs default_rng).
4. **Duplication massive** : TestDerivDataCollector, TestRiskManager, TestHMMPredictor etc. dupliqués dans `tests/` ET `deriv/tests/`.

### Architecture
1. **Dashboard** : 442 lignes en script au niveau module (pas de `if __name__`), CSS/HTML/métiers mélangés, asyncio.run dans Streamlit, import direct de Deriv/RiskManager/HMM/Kalman.
2. **Stratégies** : 2 DTO TradeSignal incompatibles, validation absente sur barrière/duration/symbol, duplication over_under/digit_diff quasi-complète.
3. **Configuration** : 3+ versions de chaque constante partagée, .env et .env.example désynchronisés, pas de pytest.ini.

### Coverage gaps critiques
1. `deriv/bot.py` (DerivBot entier) — 0 tests
2. `deriv/journal_learner.py` — 0 tests
3. `deriv/forward_demo.py` — 0 tests
4. `deriv/trading_journal.py:export_for_ml()` — 0 tests
5. `deriv/strategies/over_under.py` — 0 tests
6. `deriv/strategies/digit_diff.py` — 0 tests
7. `deriv/cpcv_validator.py` — test incomplet

---

## PLAN D'ACTION — PAR ORDRE DE PRIORITÉ

### Phase 0 — STOP THE BLEEDING (cette semaine)
1. **Supprimer** `tests/_*.py` → déplacer dans `scripts/` ou `debug/` (ne sont PAS des tests)
2. **Marquer** `test_connect_and_ping` et `test_get_ticks_live` comme `@pytest.mark.integration` + `@pytest.mark.skipif(no_network)`
3. **Ajouter** `monkeypatch.delenv` en fin de `test_max_stake_protection`
4. **Ajouter** `tests/conftest.py` qui patche TOUS les `MODELS_DIR` + `STATE_FILE` + `DEFAULT_STATE_FILE`
5. **Supprimer** `_DEFAULT_THR_UP/DOWN` mort (ensemble_predictor:529-530), `_optimal_threshold` alias (ensemble:602-609), `subscribe` mort (data_collector:198-209)
6. **Corriger** `deriv/tests/test_ensemble_robuste.py:1295` — test tronqué, pas d'assertion

### Phase 1 — CENTRALISER (semaine 2)
7. **Créer** `deriv/constants.py` — BREAKEVEN, MAX_VOTES_TO_TRADE, CRASH_SEUIL_FRAC, BREAKEVEN_WINRATE → une seule source de vérité
8. **Créer** `deriv/voting.py` — vote_to_proba(), proba_to_vote(), seuils → éliminer 4 duplications
9. **Créer** `deriv/threshold_calibrator.py` — calibrate_threshold() → éliminer duplication calib_global vs pooled
10. **Corriger** `.env.example` → ajouter 8 champs manquants + aligner avec .env
11. **Ajouter** `pytest.ini` avec testpaths, markers, filterwarnings
12. **Ajouter** `tests/__init__.py`

### Phase 2 — DÉCOUPER (semaine 3-4)
13. **Scinder** `WalkForwardOptimizer` en FoldPlanner + FoldEvaluator + PooledCalibrator + WFOReport
14. **Scinder** `EnsemblePredictor` en VotingEngine + ModelRegistry + EnsemblePredictor (coordinateur)
15. **Extraire** les 5 feature builders de `build_features()` en fonctions séparées
16. **Extraire** les 3 modes de label de `build_labels()` en fonctions séparées + vectoriser TB
17. **Scinder** `RSIDivergencePredictor.predict()` en 4 méthodes <25 lignes
18. **Scinder** `DerivDataCollector` en CandleBuilder + HistoryFetcher + DerivDataCollector
19. **Scinder** `dashboard/app.py` en services + templates (442L → modules <100L)

### Phase 3 — TESTS (semaine 5-6)
20. **Écrire** tests pour : DerivBot, JournalLearner, trading_journal.export_for_ml(), OnlineLearner.drift_detection
21. **Remplacer** AST `inspect.getsource` par tests de comportement (spy sur scaler.transform)
22. **Supprimer** `__new__` bypass dans `make_ensemble` → utiliser MagicMock
23. **Normaliser** `generate_synthetic_ohlcv` en une seule fixture conftest.py
24. **Éliminer** duplication RiskManager (3 versions → 1)
25. **Supprimer** duplication TestDerivDataCollector (2 versions → 1)

### Phase 4 — QUALITÉ (semaine 7+)
26. **Corriger** forward demo pour aligner Online vote avec le contrat 4/5 officiel
27. **Aligner** target d'entraînement et de backtest (contrat unique)
28. **Protéger** les modèles de production contre l'écrasement par le backtest
29. **Écrire** tests pour les stratégies non testées (over_under, digit_diff, regime_momentum transitions)
30. **Ajouter** pytest.ini avec testpaths, markers, filtres de warnings

---

## STATISTIQUES

- **280+** problèmes identifiés (14 CRITIQUE, 55 MAJEUR, 80+ MINEUR)
- **4** God Classes (>200 lignes)
- **13** fonctions >50 lignes
- **10+** duplications de code entre modules
- **8** modules sans aucun test
- **4** constantes dupliquées pour BREAKEVEN seul
- **347** tests passent actuellement (vs 346 avant correction FEATURE_COLS)
- **1** test FAUX positif corrigé cette session
- **~3h** pour le forward demo CRASH500 (en cours, ~170min restantes)

---

## PROCHAINES ÉTAPES IMMÉDIATES
1. Le forward demo CRASH500 tourne toujours (fin ~19:04) → vérifier le rapport final
2. BOOM500 Rise/Fall (62.5% WR, 104 trades) et XAUUSD H4 Kalman (57.9%, σ=6.8%) → prochains candidats à tester
3. Commencer Phase 0 du plan d'action dès que le demo se termine
