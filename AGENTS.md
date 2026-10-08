# AGENTS.md — Conventions du bot Deriv (C:\BOT-TRADING)

## Target (labels d'entraînement / évaluation WFO)

- **Avant** : `target = close[t+1] > close[t]` (target triviale UP/DOWN,
  mode `build_labels(df)` = `mode="sign"`).
  - Biaisé par la dérive de l'indice : sur CRASH500 M1 (+0.012 %/min),
  89.8 % des labels valent 1 → le WR mesurait la dérive, pas une compétence
  (audit 90 % : acc réelle 70.9 % = acc permutée 71.1 % = baseline UP 71.1 %).

- **Après** (P0-v2, défaut partout) : `target = triple-barrière symétrique`
  via `build_labels(df, mode="triple_barrier", touch_mult=1.5, horizon=3)`
  (équivalent `atr_mult=1.5, horizon=3`) :
  - label **1** si le prix touche **+1.5×ATR** avant -1.5×ATR dans les
  `horizon` bougies ;
  - label **0** si le prix touche **-1.5×ATR** avant +1.5×ATR ;
  - label **éliminé** (pas de trade) si ni l'une ni l'autre barrière n'est
  touchée dans l'horizon.
  - Les deux barrières sont symétriques → le biais de drift disparaît
  (baseline mesurée : 0.513 sur frxEURUSD H4 vs 0.898 en target triviale).

- **Où c'est appliqué** :
  - `deriv/ensemble_predictor.py` : `XGBoostPredictor.train()` et
    `LSTMPredictor.train()` (labels d'entraînement) ;
  - `deriv/walk_forward_optimizer.py` : `WalkForwardOptimizer.run()`
    (défaut `target_mode="triple_barrier"`), `_evaluate()` (labels
    out-of-sample via `build_labels`, baseline dérive conservée à part dans
    `up_rate`), CLI `--target {triple_barrier,sign,crash}`
    (défaut `triple_barrier`) ;
  - `deriv/tests/test_anti_leakage.py` :
    `TestTripleBarriereAntiDrift` (ratio 0.35–0.65 sur dataset drifté).

- **Legacy** : `mode="sign"` (close[t+1] > close[t]) reste disponible pour
  compatibilité (`--target sign`), `mode="crash"` pour la cible asymétrique
  CRASH500 (P(crash)).

## Quick Start

1. Installer les dépendances : `pip install -r requirements.txt`
2. Configurer `.env` (voir `.env.example`)
3. Lancer les tests : `pytest tests/ deriv/tests/`
4. Lancer le bot : `python deriv/bot_executor.py`
5. Lancer le dashboard : `streamlit run dashboard/app.py`

## Architecture

- `deriv/client.py` : Connexion WebSocket Deriv
- `deriv/risk_manager.py` : Gestion du capital et des règles de risque
- `deriv/bot_executor.py` : Orchestrateur principal (collecte -> ensemble -> risk -> execution)
- `deriv/ensemble_predictor.py` : Croisement 5 modèles (HMM, XGBoost, LSTM, Kalman, RSI)
- `deriv/online_learner.py` : Apprentissage online (River)
- `deriv/weekly_retrain.py` : Ré-entraînement hebdomadaire
- `deriv/strategies/` : Stratégies (rise_fall, boom_crash_drift, regime_momentum, over_under, digit_diff)
- `dashboard/app.py` : Dashboard Streamlit
- `forward/` : Forward demo et analyse

## Configuration

Variables .env principales :
- DERIV_APP_ID, DERIV_API_TOKEN, DERIV_SYMBOL, DERIV_DUR, DERIV_ACCOUNT_TYPE
- CAPITAL_USD, AUTO_STAKE, MAX_RISK_PCT, DAILY_STOP_PCT, MAX_CONSECUTIVE
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- RUN_NETWORK_TESTS=0 (mettre à 1 pour activer les tests réseau)
