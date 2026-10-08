# CLAUDE.md - Guide de Transmission & Conventions du Projet

> **Ce fichier sert de point d'entrée et de boussole pour Claude.**
> Il synthétise l'architecture du projet, l'assainissement récent, les conventions non négociables et les tâches en cours.

---

## 📌 1. Vue d'ensemble du projet

Ce projet est un écosystème de **bot de trading algorithmique et quantitatif** multi-marchés (Deriv Synthetics, Forex, Crypto) orienté Machine Learning :
- **Deriv (Synthetics & Forex)** : Cœur opérationnel actuel (`deriv/`).
- **Intelligence Artificielle & ML** : HMM (Hidden Markov Models), XGBoost, LSTM, filtres de Kalman, et vote d'ensemble (règle 4/5).
- **Validation Statistique** : Walk-Forward Optimization (WFO), CPCV (Combinatorial Purged Cross-Validation), détection de data leakage.
- **Gestion du Risque** : `RiskManager` dynamique, Safe Stake, Daily Stop, arrêt après séries de pertes.
- **Dashboard & Journal** : Interface Streamlit (`dashboard/app.py`), journal SQLite (`trading_journal.db`), alertes Telegram.

---

## 🧹 2. État d'assainissement récent (Octobre 2026)

Le projet a fait l'objet d'un grand nettoyage pour éliminer la dette technique et la pollution générée par des exécutions antérieures :
1. **Racine épurée** : Réduite de 140 fichiers à 12 fichiers vitaux.
2. **Centralisation des logs** : Tous les `*.txt` d'exécution ont été déplacés dans `logs/` (ignoré par Git).
3. **Rapports d'audit rangés** : Déplacés dans `docs/reports/` (`AUDIT_COMPLET.md`, `PHASE*_REPORT.md`, etc.).
4. **Scripts déplacés** : Les scripts de tests/diagnostics autonomes (`measure_crash.py`, `deep_wfo_chain.py`, etc.) sont dans `scripts/`.
5. **Fichiers déchets supprimés** : Élimination des scripts jetables (`write_test.py`, `gen_client.py`), verrous `.lock` orphelins, caches compilés et erreurs de redirection.
6. **Git & Sécurité** : `.gitignore` renforcé pour protéger `.env`, les bases SQLite, les logs, les modèles et les dossiers tiers.

---

## 📐 3. Architecture du Répertoire

```text
C:\BOT-TRADING\
├── CLAUDE.md             # Ce guide de transmission
├── AGENTS.md             # Conventions théoriques et techniques détaillées
├── TASKS.md              # Suivi complet de la roadmap et des tâches
├── RESEARCH.md           # État de l'art, sources académiques et papers
├── main.py               # Point d'entrée de lancement du bot
├── requirements.txt      # Dépendances du projet Python
├── pytest.ini            # Configuration des tests unitaires
├── .env                  # Configuration locale (JAMAIS COMMITTÉ)
├── .env.example          # Gabarit des variables d'environnement
├── state.json            # Suivi machine de l'état des phases
│
├── deriv/                # Module Deriv (Moteur principal)
│   ├── bot.py            # Classe DerivBot (orchestration)
│   ├── bot_executor.py   # Exécution des ordres & cycle en direct
│   ├── ensemble_predictor.py # Modèle d'ensemble (XGB, LSTM, HMM...)
│   ├── walk_forward_optimizer.py # Optimiseur WFO
│   ├── risk_manager.py   # Contrôle du risque et taille de mise
│   ├── strategies/       # Stratégies de trading (drift, momentum...)
│   └── tests/            # Tests spécifiques du module deriv
│
├── intelligence/         # Modèles de régimes de marché (HMM, etc.)
├── crypto/               # Stratégies et connecteurs crypto (HMMRegime...)
├── dashboard/            # Interface Streamlit (app.py)
├── forward/              # Sessions et résultats de forward testing
├── backtests/            # Résultats de WFO et classements CSV
├── scripts/              # Diagnostics, tests isolés, scripts batch (.bat)
├── logs/                 # Tous les fichiers de logs et sorties de console
├── docs/reports/         # Rapports d'audits et historiques de versions
└── tests/                # Tests unitaires transversaux
```

---

## ⚠️ 4. Règles & Conventions Critiques pour Claude

### 4.1. Cible Anti-Biais de Dérive (Triple-Barrière) — INDISPENSABLE
- **Ne JAMAIS utiliser** de target naïve `close[t+1] > close[t]` (mode `"sign"`) sur les indices synthétiques dérivants (comme `CRASH500 M1` où 89.8% des bougies sont haussières par construction). Cela crée une illusion de compétence (le modèle prédit toujours UP).
- **Toujours utiliser** le mode **triple-barrière symétrique** :
  ```python
  build_labels(df, mode="triple_barrier", touch_mult=1.5, horizon=3)
  ```
  Le label 1 correspond à +1.5 ATR touché avant -1.5 ATR, 0 si -1.5 ATR touché d'abord, et les bougies sans contact sont exclues.
  *(Consulter `AGENTS.md` pour les détails complets)*.

### 4.2. Ne jamais polluer la racine
- **Tout log** doit être écrit dans `logs/`.
- **Tout script utilitaire ou expérimental** doit être placé dans `scripts/`.
- **Aucun fichier temporaire** `.txt`, `.tmp`, ou micro-script de test ne doit rester à la racine.

### 4.3. Sécurité & Sérialisation
- **SafeUnpickler** : Ne jamais utiliser un `pickle.load()` brut non sécurisé. Toujours utiliser le chargeur sécurisé whitelisté du projet.
- **Variables d'environnement** : Ne jamais stocker de jetons API ou mots de passe dans le code. Toujours passer par la classe `Config` ou `.env`.

---

## 🚀 5. Commandes usuelles

### Exécuter les tests unitaires
```powershell
.venv\Scripts\python.exe -m pytest tests/ deriv/tests/ -q --tb=short
```

### Lancer le bot Deriv (mode dry-run sécurisé par défaut)
```powershell
.venv\Scripts\python.exe main.py deriv
```

### Lancer le dashboard Streamlit
```powershell
.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

### Lancer une optimisation Walk-Forward (WFO)
```powershell
.venv\Scripts\python.exe deriv/walk_forward_optimizer.py --symbol frxEURUSD --target triple_barrier
```

---

## 📋 6. Prochaines étapes / Roadmap en cours

Se référer en priorité à `TASKS.md` et `state.json` :
1. **Forward Demo & Calibrage Réel** : Amélioration de la stratégie sur Forex réels (`frxEURUSD`, `frxUSDJPY`, `frxEURGBP`) et indices de volatilité (`R_50`, `R_75`), car le test sur `CRASH500 M1` a révélé l'absence d'edge exploitable en raison du spread et du drift asymétrique.
2. **Dashboard Streamlit** : Poursuivre le monitoring en direct des positions, des métriques de régimes HMM et de la synchronisation avec `trading_journal.db`.
3. **Validation Anti-Leakage continue** : S'assurer que chaque nouvelle feature passe la validation `deriv/tests/test_anti_leakage.py`.
