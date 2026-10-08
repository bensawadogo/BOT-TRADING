# TASKS — Bot Deriv professionnel (mission 4/5)

Architecture : **HMM (Baum-Welch + Forward-Backward + Viterbi) → LSTM → XGBoost → TrendStrength → Momentum → Kalman → RSI → Vote → Décision**

> **RÈGLE ABSOLUE** : un trade ne se place QUE si au moins **4 modèles sur 5** sont
> d'accord. En dessous de 4/5 → **HOLD**. Toujours. (`MIN_VOTES_TO_TRADE = 4`)

## Audit target (14/09/2026) — CONCLUSION DÉFINITIVE
- WR 89-90% WFO CRASH500 = ARTIFACT (drift + target triviale)
- Preuve : acc réelle 70.9% = acc permutée 71.1% = baseline UP 71.1%
- Fix appliqué : target → triple-barrière symétrique (élimine le drift)
- Prochain test : WFO frxEURUSD H4 avec triple-barrière → verdict réel
- Bot reste en HOLD jusqu'à p-value < 0.05 sur symbole sans drift
- Permutation frxEURUSD H4 triple-barrière (1000 bougies, 100 perm.) :
  acc réelle 0.481 ≈ acc permutée 0.492, baseline TB 0.513, p-value 0.604
  → Signal réel : NON (méthodologie validée, baseline TB ≈ 50/50)
- Verdict réel WFO frxEURUSD H4 triple-barrière (15 folds, 1000 bougies,
  `backtests/wfo_summary_frxEURUSD_14400.json`) : AUCUN modèle demo_ready —
  WR bruts (HMM 63.5 %, RSI 64.3 %) invalidés par balanced WR ≤ 50 %
  (biais taux de base). → HOLD confirmé sur symbole sans drift.
- Check anti-fuite post-TB (`backtests/leakage_check_tb.txt`, script
  `backtests/leakage_check_tb.py`) : garde-fous pytest 19/19 ✔ ;
  frxEURUSD H4 → ✅ 0 fuite ; purge/barrières WFO → aucune barrière
  train n'atteint le test (marge min 31 bougies, 7 folds) ; CRASH500 M1
  → 1 flag heuristique `log_ret` (ratio 1.55x > 1.5, p=0) = FRAÎCHEUR,
  pas une fuite : le label TB de t n'évalue les barrières que sur
  [t+1, t+3] (jamais t), les features à la décision n'utilisent que
  des données ≤ t, et un spike à t gonfle ATR[t] → dépendance d'état
  légitime (sur H4, même code → ratio ~1.0, aucun flag).
- TEST DÉFINITIF CRASH500 M1 10 000 bougies, 95 folds, triple-barrière
  (`backtests/wfo_crash500_m1_tb10k.txt`, `wfo_summary_CRASH500_60.json`
  16/09/2026 02:46) : TOUS les modèles ❌ BIAIS DE TAUX DE BASE, aucun
  demo_ready. WR bruts 64.5–81.6 % = copie de la dérive (edge vs base
  UP : −7.7 % à +4.6 %, tous < 5 %). Balanced accuracy 50.5–56.7 % :
  XGBoost 51.2 %, HMM 51.7 %, Kalman 53.8 %, RSI 50.5 %, TrendStrength
  50.6 %, Momentum 56.7 % (edge +4.64 % < 5 %, σ folds 17 % > 15 %
  overfit suspecté — seul cas limite), Ensemble 52.1 %. La dérive se
  manifeste DANS les labels TB (UP réel 72–78 % : les barrières hautes
  sont touchées plus souvent sur un indice dérivant). p-values vs
  hasard 0.0000 = significativité du drift copié, pas d'un skill.
  → CONCLUSION : ces marchés ne sont pas prévisibles avec cette
  approche (features techniques + XGB/HMM/Kalman/heuristiques). HOLD
  devient une conclusion scientifique fermée, pas une prudence.
  Seule piste non tranchée : Momentum bal 56.7 % (à valider par test
  de permutation stratifié par classe si l'on veut le poursuivre).


## 📦 Fichiers livrés

| Fichier | Rôle |
|---|---|
| `deriv/data_collector.py` | Collecte ticks temps réel + historique, agrège les bougies OHLCV |
| `deriv/ensemble_predictor.py` | 7 modèles (HMM, XGBoost, LSTM, Kalman, RSI, TrendStrength, Momentum) + vote majoritaire qualifié |
| `deriv/market_structure.py` | Structure de marché (swing highs/lows, zones demand/supply) |
| `deriv/bot_executor.py` | Boucle analyse → risk → exécution CALL/PUT + alertes Telegram + **ré-entraînement hebdo auto** |
| `deriv/train_and_backtest.py` | Entraînement + backtest out-of-sample (breakeven 55.6%) |
| `deriv/walk_forward_optimizer.py` | **Walk-Forward Optimization** (purge 1 + embargo, Wilson, p-value, overfitting, verdict démo) |
| `deriv/leakage_detector.py` | **Détection auto de feature leakage** (méthode López de Prado, rapport CLI) |
| `deriv/symbol_optimizer.py` | **Diagnostic automatique win rate par symbole × timeframe** (classement WFO) |
| `deriv/weekly_retrain.py` | **Ré-entraînement hebdomadaire automatique** (état JSON, CLI --status/--force/--daemon) |
| `deriv/risk_manager.py` | Couche risque étendue (`check`, `record`, `stats`, `TradeResult`, circuit breaker, **verrou breakeven 55.6%**) |
| `conftest.py` | Isolation pytest : les tests n'écrasent JAMAIS les modèles de production |
| `.env` / `.env.example` | Configuration (demo par défaut, jamais de real sans validation) |

## 🧠 Modèles actifs (7 au total)

| Modèle | Statut | Source d'inspiration |
|---|---|---|
| HMM (Baum-Welch + Viterbi) | ✅ actif | `intelligence/hmm_regime.py` |
| XGBoost | ✅ actif | `Stock-Prediction-Models` (MIT) |
| LSTM | ⚠️ neutre | tensorflow manquant (Python 3.14 incompatible) |
| Kalman Momentum | ✅ actif | `intelligence/kalman_filter.py` |
| RSI Divergence | ✅ actif | `deriv/ensemble_predictor.py` |
| TrendStrength (EMA+ADX) | ✅ actif | `OmashelCap` (MIT) |
| Momentum (ROC) | ✅ actif | `OmashelCap` (MIT) |

> **Vote** : seuils à 0.7 (BUY) / 0.3 (SELL). LSTM vote neutre (0.5) tant que tensorflow absent.
> Avec 6 modèles actifs + LSTM neutre, la règle 4/5 reste pleinement opérationnelle.

## ✅ Tests et validation

- **233 tests pytest** (40 existants + 193 nouveaux) : `python -m pytest tests/ deriv/tests/ -q`
  → **233 passed en 105s (12/09/2026)** — dont **17 tests anti-leakage** (`deriv/tests/test_anti_leakage.py`)
  - `deriv/tests/test_ensemble_robuste.py` : 163 collectés (dont 11 cas
    paramétrés règle 4/5 — 16 classes : features (8 cols), HMM, XGB, LSTM-fallback,
    Kalman, RSI, TrendStrength, Momentum, **règle 4/5 sur 7 modèles**,
    walk-forward, **timeframes supérieurs H1/H4/M15 (API v4)**, **statistiques**
    Wilson/p-value/overfitting, risk manager, collecteur, symbol_optimizer,
    ré-entraînement hebdo)
  - `tests/` racine : 53 tests existants (diagnostics 4/5, intelligence, Deriv)
  - `conftest.py` racine : chaque test écrit dans un dossier temporaire →
    les pickles de production ne sont jamais écrasés par les tests
- **Robustness test** : 6/6 scénarios corrects (FORT_HAUT→BUY, FORT_BAS→SELL,
  RANGE→HOLD, HAUT_FAIBLE→HOLD, BAS_FAIBLE→HOLD, BRUIT→HOLD)
- **Règle 4/5** : testée et verrouillée (ne jamais baisser à 3)

## 🛡️ Garde-fous anti-leakage — López de Prado (2018) (12/09/2026)

Les 5 garde-fous sont implémentés et validés (`deriv/tests/test_anti_leakage.py`, 17 tests) :

| # | Garde-fou | Implémentation | Résultat |
|---|---|---|---|
| 1 | Scaler fit sur train uniquement | XGBoost + HMM + LSTM : split 80/20 **avant** `fit` ; val/live = `transform()` seul | ✅ testé (center vs mediane train) |
| 2 | Target avant filtrage | Architecture **découplée** : `build_labels()` séparé, dernière ligne exclue (`iloc[:-1]`) | ✅ 0 NaN, alignement recalculé ligne à ligne |
| 3 | Purge + Embargo | WFO : `PURGE=1` (label train[-1] pointe vers test) + embargo, assert anti-chevauchement | ✅ 0 chevauchement sur tous les folds |
| 4 | Détection feature leakage | `deriv/leakage_detector.py` : corr(t0) vs corr(t-1), ratio > 1.5 + p < 0.01 | ✅ **0 fuite sur données réelles H4** (8 features) ; fuite artificielle détectée |
| 5 | Stabilité inter-seeds (proxy PBO) | WR sur 4 seeds → std < 15 pts | ✅ test passé (std < 15%) |
| — | Garde-fou au démarrage | `bot_executor._run_startup_checks()` : leakage + alignement target, stop + Telegram si fuite | ✅ branché dans `start()` |

**Impact majeur du purge (garde-fou 3)** : re-run WFO H4 EURUSD 5000 bougies
(1557 chargées, 26 folds, purge+embargo) → **XGBoost retombe à 46.7%**
(std 5.4%, 45 trades, écart in/out 20 pts). Le 60.5% du 11/09 était donc
**partiellement un artefact de frontière train/test non purgée**.
Aucun modèle ne dépasse 55.6% sur H4 → **mode historique maintenu**.

## 📈 Walk-Forward Optimization — résultats réels (2026-09-11)

### 1. R_75 / M1 / 3000 bougies / 25 folds

`python deriv/walk_forward_optimizer.py --symbol R_75 --count 3000`
CSV + JSON détaillés dans `backtests/`.

| Modèle | WR WFO | Trades | σ folds | In-sample | Overfit | p-value |
|---|---|---|---|---|---|---|
| XGBoost | 51.8% | 459 | 12.1% | **86.9%** | ⚠️ oui | 0.23 |
| HMM | 49.6% | 655 | 9.7% | 52.6% | non | 0.59 |
| Kalman | 49.4% | 747 | 10.3% | — | non | 0.64 |
| Momentum | 48.0% | 731 | 8.8% | — | non | 0.87 |
| RSI | 47.8% | 113 | 29.0% | — | ⚠️ oui | 0.71 |
| Ensemble 4/5 | 27.5% | 40 | 28.7% | 60.0% | ⚠️ oui | 0.998 |
| TrendStrength | 39.1% | 23 | 8.2% | — | ⚠️ oui | 0.90 |

### 2. EUR/USD H1 / 2695 bougies paginées / 62 folds (train 300, embargo 10, test 75)

`python deriv/walk_forward_optimizer.py --symbol frxEURUSD --granularity 3600 --count 5000`

| Modèle | WR WFO | Trades | σ folds | In-sample | Overfit | p-value |
|---|---|---|---|---|---|---|
| RSI | **56.4%** | 133 | 34.9% | — | ⚠️ instable | 0.083 |
| Ensemble 4/5 | **58.6%** | 29 | 42.4% | 72.2% | ⚠️ éch. faible | 0.229 |
| TrendStrength | 54.5% | 22 | 21.7% | — | ⚠️ instable | 0.416 |
| HMM | 50.8% | 543 | 17.8% | 47.8% | ⚠️ instable | 0.366 |
| XGBoost | 49.3% | 402 | 23.6% | **74.9%** | ⚠️ oui (25.6 pts) | 0.636 |
| Momentum | 48.8% | 644 | 17.7% | — | ⚠️ instable | 0.749 |
| Kalman | 46.1% | 679 | 14.2% | — | non | 0.981 |

**Re-validation 12/09/2026** (fenêtre glissée — 62 folds, mêmes paramètres) :

| Modèle | WR WFO | Trades | σ folds | In-sample | Overfit | p-value |
|---|---|---|---|---|---|---|
| TrendStrength | 62.5% | 24 | 11.4% | — | ⚠️ éch. faible | 0.154 |
| Ensemble 4/5 | 57.6% | 33 | 40.1% | 50.0% | ⚠️ instable | 0.243 |
| RSI | 54.9% | 133 | 32.3% | — | ⚠️ instable | 0.149 |
| HMM | 52.6% | 487 | 18.1% | 46.6% | ⚠️ instable | 0.138 |
| XGBoost | 51.1% | 282 | 28.2% | **76.8%** | ⚠️ oui (25.8 pts) | 0.383 |
| Momentum | 49.4% | 644 | 17.6% | — | ⚠️ instable | 0.639 |
| Kalman | 45.5% | 681 | 14.1% | — | non | 0.991 |

> La re-validation confirme l'instabilité H1 : le RSI qui dépassait 56.4% le
> 11/09 retombe à 54.9% sur la fenêtre glissée. **Aucun modèle H1 n'est
> `demo_ready`** — le WR H1 n'est pas reproductible d'une fenêtre à l'autre.

### 3. EUR/USD H4 / 1559 bougies paginées (1 an) / 26 folds (train 200, embargo 5, test 75)

`python deriv/walk_forward_optimizer.py --symbol frxEURUSD --granularity 14400 --count 5000`

| Modèle | WR WFO | Trades | σ folds | In-sample | Overfit | p-value |
|---|---|---|---|---|---|---|
| XGBoost | **60.5%** | **177** | 24.8% | 40.0% | **0% overfit** | **0.0033** ✅ |
| Momentum | 54.2% | 264 | 15.5% | — | ⚠️ instable | 0.098 |
| HMM | 53.1% | 196 | 16.9% | 31.2% | ⚠️ instable | 0.216 |
| Kalman | 45.7% | 293 | 15.1% | — | ⚠️ instable | 0.936 |
| TrendStrength | 42.9% | 14 | 31.2% | — | ⚠️ éch. faible | 0.788 |
| Ensemble 4/5 | 41.2% | 17 | 27.8% | 0.0% | ⚠️ éch. faible | 0.834 |
| RSI | 37.2% | 51 | 29.3% | — | ⚠️ instable | 0.976 |

### 4. Scan complet 10 combinaisons (`deriv/symbol_optimizer.py`)

Résultats consolidés dans `backtests/classement_symbols.csv` :

1. `R_75 H1` : WR = 100.0% (Ensemble 4/5, 2 trades, σ = 0.0%)
2. `EURUSD H4` : WR = **72.0%** (XGBoost, 25 trades, σ = 22.6%)
3. `BTCUSD M15` : WR = **66.7%** (TrendStrength, 9 trades, σ = 0.0%)
4. `EURUSD H1` : WR = **61.9%** (RSI, 21 trades, σ = 3.3%)
5. `EURUSD M1` : WR = **59.3%** (XGBoost, 499 trades, σ = 10.5%)
6. `R_75 M1` : WR = **57.0%** (RSI, 135 trades, σ = 27.0%)
7. `BTCUSD H1` : WR = 55.6% (RSI, 18 trades, σ = 42.4%)
8. `BTCUSD M5` : WR = 53.8% (XGBoost, 303 trades, σ = 11.8%)
9. `EURUSD M5` : WR = 52.5% (Kalman, 444 trades, σ = 9.0%)
10. `EURUSD M15` : WR = 50.5% (XGBoost, 313 trades, σ = 9.4%)

**Verdict Passage en Démo Réelle :**

- **Critères stricts** : WR $\ge 55.6\%$ ET $\sigma < 10\%$ ET $\ge 200$ trades ET pas d'overfitting.
- **Résultat Pagination H4** : La pagination epoch a résolu le manque de trades en passant de 25 trades à **177 trades** sur 1 an complet (26 folds). XGBoost H4 confirme un solide **60.5% de Win Rate** avec une p-value de **0.0033** (significativité statistique à 99.7%).
- **Statut 200 trades** : 177 trades reste légèrement sous le seuil strict de 200 trades (car l'API Deriv limite l'historique H4 à 1559 bougies sur ce compte/app_id).
- **Décision** : Le bot **reste en mode historique rafraîchi / HOLD** par rigueur méthodologique jusqu'à confirmation sur 200+ trades ou détente calibrée du seuil.

### ✅ Checklist pagination H4 (2026-09-11) — COMPLÈTE

- [x] `get_candle_history_full()` ajoutée avec pagination epoch (`deriv/data_collector.py` L257)
- [x] `walk_forward_optimizer.py` mis à jour → appel `collector.get_candle_history_full(target_count=args.count)`
- [x] `symbol_optimizer.py` mis à jour → appel `collector.get_candle_history_full(target_count=count)`
- [x] Test `test_pagination_produit_plus_de_bougies` ajouté — **11/11 tests passés** (`TestTimeframesSupérieurs`)
- [x] WFO H4 re-lancé avec `--count 5000` (2026-09-11 19:54→19:56)
- [x] Résultat confirmé : **n_trades = 177** (< 200 — API Deriv limite H4 à ~1559 bougies)
- [x] XGBoost H4 : **60.5% WR**, p-value **0.0033** ✅, IC Wilson [53.1% ; 67.4%]
- [ ] Démo réelle : **HOLD** — seuil 200 trades non atteint (177/200), σ = 24.8% > 10%

### ✅ Checklist H1 / tests (2026-09-12) — COMPLÈTE

- [x] `data_collector.py` API v4 : `subscribe: 0` + `adjust_start_time: 1`
  (requête ponctuelle, champs `history.prices`/`history.times` requis)
- [x] `WalkForwardOptimizer(granularity=...)` adaptatif : H4 (train 200/test 50/
  embargo 5), H1 (train 300/test 75/embargo 10), M1-M15 (train 400/test 100/embargo 30)
- [x] `symbol_optimizer.py` : 10 combinaisons dont H1 (frxEURUSD, cryBTCUSD, R_75) + H4
- [x] `TestTimeframesSupérieurs` : **11/11 passés** (WFO adaptatif, granularité 3600,
  structure requête API v4 `subscribe`/`adjust_start_time`)
- [x] **216/216 tests passent** (12/09/2026, 102s)
- [x] WFO H1 5000 lancé : 62 folds, **aucun modèle `demo_ready`** (RSI retombe à 54.9%)
- [x] WFO H4 500 lancé : sous le seuil sur la fenêtre courte ; le verdict H4 de
  référence reste le run paginé du 11/09 (XGBoost 60.5%, p = 0.0033, 177 trades)
- [ ] Démo réelle : **HOLD** — H1 instable entre fenêtres, H4 sous le seuil de trades

### ✅ Checklist Garde-fous anti-leakage — López de Prado (2026-09-12) — COMPLÈTE

- [x] **Garde-fou 1 (scaler)** : fit UNIQUEMENT sur le train 80% — corrigé dans
  `XGBoostPredictor.train`, `HMMPredictor.train` (fit train → transform séquence
  complète, scaler persisté dans le pickle), `LSTMPredictor.train`
- [x] **Garde-fou 2 (target avant filtrage)** : architecture découplée vérifiée —
  `build_labels()` (shift(-1) puis exclusion de la dernière ligne) + helper
  `feat_avec_target()` ; test de ré-alignement ligne à ligne sans fuite
- [x] **Garde-fou 3 (purge + embargo)** : `WalkForwardOptimizer.PURGE = 1` —
  la dernière bougie du train (dont la target pointe vers le test) est retirée ;
  `folds_plan()` prend le paramètre purge ; schéma
  `[TRAIN][PURGE 1b][EMBARGO][TEST]`
- [x] **Garde-fou 4 (détection leakage)** : `deriv/leakage_detector.py` —
  corr(feature[t], target) vs corr(feature[t-1], target), ratio > 1.5 + p < 0.01 ;
  détecte une feature fuitée plantée (test `test_feature_future_detectee`) ;
  0 faux positif sur les 8 features réelles (dataset marche aléatoire)
- [x] **Garde-fou 5 (stabilité inter-seeds)** : std des win rates sur 4 seeds
  < 15% → **passé** (proxy PBO simplifié)
- [x] **Garde-fou au démarrage du bot** : `bot_executor._run_startup_checks()` —
  bloque le trading si fuite détectée (alerte Telegram) ; target alignée injectée
  dans le détecteur
- [x] **17/17 tests `test_anti_leakage.py` passent** — synthétique en marche
  aléatoire (cumsum) : les incréments i.i.d. garantissent corr(r_t, target_t)=0
  sans fuite (un niveau + bruit i.i.d. crée une autocorrélation -0.5 structurelle
  → 6 faux positifs, corrigé)
- [x] **233/233 tests** — zéro régression des corrections scaler/purge

## 🚀 Démarrage rapide

```bash
# 1. TESTS COMPLETS (233 tests, ~2 min)
.venv\Scripts\python.exe -m pytest tests/ deriv/tests/ -q

# 1bis. Garde-fous anti-leakage uniquement
.venv\Scripts\python.exe -m pytest deriv/tests/test_anti_leakage.py -v

# 2. Walk-Forward Optimization (référence statistique — REMPLACE le 70/30)
.venv\Scripts\python.exe deriv\walk_forward_optimizer.py --symbol R_75 --count 3000
.venv\Scripts\python.exe deriv\walk_forward_optimizer.py --symbol frxEURUSD --granularity 3600 --count 1000
.venv\Scripts\python.exe deriv\walk_forward_optimizer.py --symbol frxEURUSD --granularity 14400 --count 500

# 3. Scan automatique de toutes les combinaisons
.venv\Scripts\python.exe deriv\symbol_optimizer.py

# 4. Ré-entraînement hebdomadaire
.venv\Scripts\python.exe deriv\weekly_retrain.py --status   # état
.venv\Scripts\python.exe deriv\weekly_retrain.py --force    # forcer

# 5. Lancer le bot en DEMO (après WFO OK et token Telegram configuré)
.venv\Scripts\python.exe -m deriv.bot_executor
```

## 📊 Derniers runs réels

- **WFO H1 EURUSD 5000** (12/09, 62 folds, ~2695 bougies paginées) : aucun modèle
  `demo_ready` — TrendStrength 62.5% (24 t), Ensemble 57.6% (33 t), RSI 54.9% (133 t).
- **WFO H4 EURUSD 5000 — RE-RUN AVEC PURGE** (12/09, 26 folds, 1557 bougies) :
  XGBoost **46.7%** (std 5.4%, 45 trades) — le 60.5% du 11/09 était un artefact
  de frontière non purgée → **candidat H4 invalidé, mode historique maintenu**.
  HMM 54.3% (σ 19.4%), RSI 41.2%, Ensemble4/5 31.2% — tous sous le seuil.
- **WFO H1 EURUSD 1000** (12/09, 9 folds) : RSI 59.1% (22 t, p 0.26 — non significatif).
- **Scan symbol_optimizer (10 combinaisons)** : `backtests/symbol_ranking.csv` à jour.
- **Entraînement final** (12/09 04:37, 1500 bougies M1 frxEURUSD réelles) :
  HMM ✅ (labels Bull/Range/Bear, scaler fit sur train 80% — garde-fou 1) et
  XGBoost ✅ (accuracy val **52.0%** — proche du hasard, aucun signe d'overfit ;
  seuils calibrés ↑0.75 ↓0.25, scaler persisté dans le pickle). LSTM ignoré
  (tensorflow indisponible sur Python 3.14). `.retrain_state.json` à jour →
  `Ré-entraînement requis : non` (scheduler hebdo armé).
- **Leak check données réelles H4** (12/09, 1000 bougies paginées) :
  **✅ 0 fuite détectée sur 8 features** — toutes temporellement propres.
- Boucle d'analyse en conditions réelles : `HOLD | Votes BUY:3 SELL:0 HOLD:2` (règle 4/5 respectée).

> **Mode de fonctionnement actuel** : sans `DERIV_API_TOKEN`, le serveur Deriv
> refuse le streaming `ticks` live mais autorise `ticks_history`. Le bot bascule
> automatiquement en **mode historique rafraîchi** avec backoff 5→30s.
> Tout est fonctionnel dès que le token est ajouté au `.env`.

## 🔬 Recherche documentaire

Fichiers dans `research/` — sources uniquement, zéro code modifié.
Voir `research/index.md` pour le tableau de bord consolidé (P0/P1/P2).

## ⛔ NE PAS TOUCHER

- `MIN_VOTES_TO_TRADE = 4` (ne jamais baisser à 3)
- `DERIV_ACCOUNT_TYPE=real` avant validation complète
- `DERIV_ALLOW_REAL=1` sauf décision explicite (le bot bloque sinon)

## ♻️ Dépendances (état de l'installation)

- `xgboost` 3.4.1 : ✅ installé (modèle 2 actif)
- `aiohttp` : ✅ installé (Telegram)
- `pywavelets` : ✅ installé
- `tensorflow` : ❌ **non installable sur Python 3.14** (aucune wheel publiée).
  Sans lui le LSTM vote neutre (0.5) ; l'ensemble reste utilisable via
  HMM+XGB+Kalman+RSI (4/5 atteignable). Pour activer le LSTM : Python 3.11–3.13
  + `pip install tensorflow`.
- Collecte = `deriv-sdk` (WebSocket officiel), alertes = `aiohttp`.

## 📌 Notes techniques

- Modèles persistés dans `deriv/models/` (gitignored) : `hmm_deriv.pkl`,
  `xgb_deriv.pkl`, `lstm_deriv.h5` + scaler, `.retrain_state.json` (hebdo).
- **Backtest de référence = Walk-Forward** (non-ancré, embargo 30 bougies,
  Wilson, p-value binomiale, détection d'overfitting) — le 70/30 reste
  dispo dans `train_and_backtest.py` mais n'est plus une preuve de robustesse.
- Le WFO désactive la sauvegarde des modèles (`_disarm_save`) → les pickles
  de production ne sont jamais écrasés par une session d'optimisation.
- Le RiskManager bloque : daily stop -5%, 3 pertes consécutives, cooldown
  15 min, **confiance < 55.6% (breakeven)**, stake > min(2% capital, auto-max).

## 🌍 Mission « vrais marchés » — scan 12 combos (12/09 12:09-12:22)

Hypothèse testée : les vrais marchés (forex/métaux/crypto) ont des patterns
humains exploitables, contrairement aux synthétiques (RNG audité). Scan via
`symbol_optimizer.py` (WFO purgé, garde-fous actifs), classement dans
`backtests/classement_symbols.csv`.

| Combo | WR | σ | Folds | Trades | Modèle | Verdict |
|---|---|---|---|---|---|---|
| EURGBP H4 | 100% | 0% | 5 | **2** | RSI | bruit (échantillon) |
| R_75 H4 | 100% | 0% | 5 | **2** | RSI | bruit (échantillon) |
| USDJPY H4 | 67.9% | 12.1% | 5 | 28 | RSI | candidat (petit n) |
| EURUSD H1 | 66.7% | 0% | 9 | **6** | TrendStrength | bruit |
| R_50 H4 | 60% | 0% | 5 | 15 | XGBoost | faible n |
| **XAUUSD H4** | **57.9%** | **6.8%** | 5 | **57** | Kalman | **meilleur profil** |
| GBPUSD H1 | 53.2% | 8.1% | 9 | 94 | HMM | sous seuil |
| GBPUSD H4 | 53.1% | 10.1% | 5 | 49 | HMM | sous seuil |
| ETHUSD H4 | 52.4% | 36.5% | 5 | 21 | RSI | instable |
| BTCUSD H4 | 51.9% | 5.1% | 5 | 52 | Kalman | sous seuil |
| XAUUSD H1 | 51.9% | 12.3% | 9 | 81 | HMM | sous seuil |
| EURUSD H4 | 50% | 30.9% | 5 | 16 | HMM | très instable |

**Lecture** : XAUUSD H4 est le seul combo au-dessus du breakeven avec un
échantillon décent (57 trades) ET une stabilité forte (σ=6.8%) — conforme à
la prédiction mission (Or = trending, bon pour le Kalman). Les 100% à 2
trades sont statistiquement vides. USDJPY H4 (67.9%, 28 trades) est
prometteur mais à confirmer (p≈0.02, limite).

**Étape 3 terminée 13/09** : 6 deep WFO (count 5000, 26 folds chacun, ~1550 bougies
H4 paginées) — chaîne `deep_wfo_chain.py` OK. **Aucun modèle `demo_ready`.**
Résultat clé : les 5 folds du scan étaient OPTIMISTES (Kalman XAUUSD
57.9% → 48.4% en 26 folds, RSI USDJPY 67.9% → 51.2%).
Anomalies : TrendStrength XAUUSD 83.3% sur **6 trades** + Kalman 0 trade
→ `No data for Trader` (circuit Breaker) dans les 2 cas = **bruit d'échantillon**,
non un edge. Verdict : **bot en HOLD, mode historique rafraîchi.**

## Stratégie Boom/Crash Drift — 13/09/2026

### Implémentation
- `deriv/strategies/boom_crash_drift.py` (191 lignes) : stratégie drift structurel.
  Détection de spike **adaptative** (3×std de la fenêtre, plancher 0.08%) —
  le seuil fixe 0.5% ne détectait RIEN sur les données réelles.
  Sur BOOM → SHORT après spike ; sur CRASH → LONG ; SL/TP par ATR (RR 1.8).
- `deriv/trading_journal.py` : `spread_cost_pct` + `pnl_net_pct` persistés,
  `_apply_exit()` calcule `pnl_net = pnl − spread`, migration idempotente des
  colonnes pour les DB existantes, `pnl_net_pct` retourné par log_exit/log_exit_contract.
- `deriv/bot_executor.py` : stratégie instanciée si symbole BOOM/CRASH,
  `spread_cost_pct=0.2` injecté dans la JournalEntry.
- `deriv/validate_boom_crash_drift.py` : validation drift sur données réelles.

### Tests : 16 nouveaux (Boom/Crash) → total 43/43 sur les 3 fichiers ciblés
Détection spike (Boom/Crash, pas de spike, <2 bougies), machine à états
(SHORT sur Boom SL>entry, LONG sur Crash SL<entry, reset après signal,
spread 0.2), journal (pnl_net = pnl − spread, négatif si spread gros,
zéro si pas de spread), TradeSignal (RR short/long = 1.8).

### Données réelles BOOM500/CRASH500 (3000 bougies M1, 13/09)
Analyse de distribution (analyse des moves par bougie) :
- **CRASH500** : moves min −0.294% / max +0.016% / std 0.039%.
  → drift haussier régulier (+0.012%/min), spikes = chutes soudaines.
  Le max +0.016% montre que la hausse n'est JAMAIS rapide → drift réel.
- **BOOM500** : symétrique (drift baissier + spikes haussiers).
- Seuil adaptatif 3×std = 0.117% → ~100 spikes détectés sur 3000 bougies.
- Seuil fixe 0.2% → 26 spikes CRASH / 39 spikes BOOM (plus marqués).

### Drift après spike (signes corrects, in-sample)
| Horizon | CRASH500 (0.2%) | BOOM500 (0.2%) |
|---|---|---|
| h=3 | 73.1% | 79.5% |
| h=5 | 73.1% | 71.8% |
| h=10 | **73.1%** | 71.8% |
| h=30 | 65.4% | 53.8% |

→ le drift existe (65-73% de signes corrects) MAIS la magnitude est faible
(~0.06-0.09% sur 10-30 min), inférieure au spread AR 0.2% sur un modèle SL/TP.
PnL net SL/TP : BOOM −15.57%, CRASH −16.27% (90/91 trades).

### Walk-Forward MODÈLE BINAIRE (Rise/Fall, direction à expiration)
Seuil 0.2% fixé a priori, horizon 10, 5 folds :
- **CRASH500 : WR 69.6% sur 23 trades / 5 folds** → au-dessus du breakeven
  Rise/Fall (~55.6%). MAIS variance énorme entre folds
  (50%, 100%, 87.5%, 100%, 40%) → **23 trades non significatifs** (Wilson ~[48-85%]).
- **BOOM500 : résultat dans le même ordre** (n petit, folds instables).

### Verdict honnête
1. **Le drift structurel est CONFIRMÉ** par les données réelles (65-73% de
   signes corrects après spike) — l'insight académique est valide.
2. **Le seuil fixe 0.5% était faux** : aucun spike réel en M1 ne l'atteint
   (max −0.294%). Le seuil adaptatif est indispensable.
3. **Modèle SL/TP : perdant net** (drift < spread AR).
4. **Modèle binaire Rise/Fall : prometteur (69.6% WR) mais échantillon
   insuffisant** (23 trades, folds instables) → ne PAS passer en démo.
5. Le bot **reste en HOLD / mode historique** jusqu'à un échantillon ≥ 100
   trades out-of-sample stable (WR ≥ 55.6% ET σ < 10%).

**Piste suivante** : augmenter l'historique (10 000+ bougies), affiner le
seuil de spike (quantile 99.5% au lieu de 3×std), et re-tester le modèle
binaire sur un échantillon ≥ 100 trades.

### Test à grande échelle — 13/09 (définitif)

`deriv/validate_boom_crash_big.py` : 10 000 bougies M1 (pagination epoch),
seuil calibré SUR LE TRAIN uniquement (quantile des |moves|, aucun paramètre
appris sur le test), WFO binaire à 6 folds, stats complètes (Wilson, binomial).

| Run | WR OOS | Trades | Folds | Wilson 95% | p-value | σ folds |
|---|---|---|---|---|---|---|
| CRASH500 q99.5 | 64.1% | 39 | 5 | [48.4–77.3%] | 0.183 | 21.1% |
| CRASH500 q99 | 63.4% | 82 | 6 | [52.6–73.0%] | 0.094 | 13.5% |
| **BOOM500 q99** | **62.5%** | **104** | **6** | **[52.9–71.2%]** | **0.093** | **18.1%** |

Détail BOOM500 (seuil auto ~0.17–0.21% par fold) :
folds = 60.0% (25t), 42.3% (26t), 81.8% (11t), 62.5% (16t), 70.0% (20t),
100% (6t) — 4/6 folds ≥ 60%.

**Lecture finale :**
1. **Le critère ≥ 100 trades est atteint (BOOM500 : 104)** et le WR tient
   à 62.5% — l'edge drift ne s'effondre PAS avec le volume (contrairement
   aux faux positifs précédents : XGBoost H4 60.5%→46.7% une fois purgé).
2. **Deux indices indépendants convergent** : BOOM 62.5% / CRASH 63.4% —
   ce n'est pas l'artefact d'un seul symbole.
3. **MAIS p = 0.093 (> 0.05) et σ = 18.1% (> 10%)** → les critères stricts
   ne sont pas remplis. À WR=62.5%, il faudrait ~200 trades pour p<0.05.
4. Wilson [52.9–71.2%] contient encore 55.6% → l'hypothèse « pas d'edge »
   n'est pas exclue à 95% de confiance.

**Verdict : edge PROBABLE mais non PROUVÉ (p=0.09).** C'est le meilleur
candidat depuis le début du projet — le seul qui tient sur 100+ trades
out-of-sample. Deux options :
- **Option prudente (recommandée)** : reste en HOLD, collecte encore
  l'historique (~20 000 bougies = ~14 jours M1) pour atteindre p<0.05.
- **Option agressive** : démo réelle 1$ sur BOOM500 Rise/Fall h=10 en
  papier-trading strict, 50 trades, en journalisant tout — le journal
  donnera la réponse en conditions réelles.

Le bot reste en HOLD par défaut. `DERIV_ALLOW_REAL` reste 0.

