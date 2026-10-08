# Rapport Problèmes — Bot Deriv (`deriv/`)

Document d'analyse exhaustive des problèmes détectés, causes racines,
correctifs appliqués et risques restants.  
État des lieux au **16/09/2026** (après correction P0-v2 + forward-demo).

---

## 1. Biais de cible (le blocage fondamental)

### P1 — Target triviale `close[t+1] > close[t]` biaisée par le drift
- **Symptôme** : WR 89–90 % en WFO sur CRASH500 M1, semblait « excellent ».
- **Cause racine** : le label `UP` copie la dérive de l'indice (+0.012 %/min →
  89.76 % de bougies haussières). Prédire UP = copier la dérive.
- **Preuve (audit 90 %)** : acc réelle **70.9 %** = acc permutée **71.1 %** =
  **baseline UP 71.1 %** → ratio signal/bruit = **1.00×**. Aucun signal.
- **Correctif** : target → **triple-barrière symétrique** `build_labels(mode="triple_barrier", touch_mult=1.5, horizon=3)`, défaut partout
  (`_evaluate`, `run`, `XGBoostPredictor.train`, `LSTMPredictor.train`, CLI `--target`).

### P6 — La triple-barrière ne cantonne PAS le drift sur actif tendanciel
- **Observation** : même TB, ratio label=1 ≈ 56–65 % (pas 50/50 idéal).
- **Cause** : sur un indice montant, la barrière **+1.5×ATR** est atteinte
  plus souvent que −1.5×ATR (le prix « tombe » dans le range haussier) → le
  déséquilibre de label reflète la tendance, pas un signal.
- **Mitigation appliquée** : la **baseline UP est mesurée à part** (`up_rate` /
  `BaselineUP` en forward) et comparée au WR modèle *bougie par bougie*.
  Seul un modèle qui **dévie de la baseline** avec p‑value < 0.05 est un edge.
