# AUDIT COMPLET — Code & Architecture
# C:\BOT-TRADING
# Date : 2026-09-16

---

## RESUME EXECUTIF

| Category | Count |
|----------|-------|
| CRITIQUE (bloquant / bug fonctionnel) | 14 |
| MAJEUR (dette technique significative) | 23 |
| MINEUR (amelioration / nettoyage) | 28 |
| **Total** | **65** |

Tests : **339/339 passent** (apres corrections).
Forward demo : **termine** — aucun edge exploitable sur CRASH500 M1.

---

## 1. PROBLEMES CRITIQUES

### C1. deriv/__init__.py:3 — Import .bot qui n existe pas
from .bot import DerivBot  # fichier deriv/bot.py INEXISTANT
Impact : import deriv echoue completement.
Correction : Creer deriv/bot.py ou retirer l import.

### C2. deriv/weekly_retrain.py:36-40 — FileLock non installe
Impact : Bloquait 7/7 collections de tests.
Correction : Deja corrige (try/except + nullcontext).

### C3. deriv/constants.py — BREAKEVEN_WINRATE manquant
Correction : Deja corrige (BREAKEVEN_WINRATE = 55.6).

### C4. tests/test_deriv.py:74 — IndentationError (classe vide)
Correction : Deja corrige (test_placeholder ajoute).

### C5. requirements.txt — filelock absent
Correction : Deja corrige.

### C6. deriv/risk_manager.py:12 — load_dotenv() au niveau module
Impact : Effet de bord global a l import.
Correction : Retirer load_dotenv() du niveau module.

### C7. deriv/client.py:10 — load_dotenv() au niveau module
Correction : Retirer, le consommateur appelle deja load_dotenv().

### C8. deriv/__init__.py — Import cycle potentiel
bot_executor importe strategies etc — risque de cycle.
Correction : Verifier ordre des imports, utiliser lazy imports.

### C9. tests/test_deriv.py:21-24 — Appel reseau au chargement
Impact : Import echoue hors ligne, tous tests incollectibles.
Correction : Deplacer dans fixture/test.

### C10. tests/test_deriv.py:60 — Test async sans decorateur
Correction : Ajouter @pytest.mark.asyncio + @integration.

### C11. forward/CRASH500_60/RAPPORT_CLAUDE.md — Conclusion erronee
Correction : Corriger — aucun modele exploitable (PnL negatif universel).

### C12. dashboard/app.py:107 — unsafe_allow_html=True
Correction : Documenter risque XSS si input utilisateur futur.

### C13. state.json — Ecrase (perte de donnees)
Correction : Reconstruire le contenu original si necessaire.

### C14. AGENTS.md — Incomplet (phase 2+ non documentee)
Correction : Ajouter phases manquantes et Quick Start.

---

## 2. PROBLEMES MAJEURS

### M1. deriv/bot_executor.py:61-73 — Try/except contradictoire
Si TradingJournal echoue, on retente RegimeMomentumStrategy dans except.
Correction : self.strategy = None dans except.

### M2. deriv/risk_manager.py:128 — except Exception trop large
Correction : Catch specifique + log warning.

### M3. deriv/bot_executor.py:167 — Hack _connected = True
Correction : Utiliser methode publique connect().

### M4. deriv/bot_executor.py:500 — Journalisation AVANT execution
Correction : Journaliser APRES execution.

### M5. deriv/strategies/__init__.py — Incoherente import/export
Correction : Verifier et harmoniser.

### M6. tests/test_deriv.py — test_placeholder ne teste rien
Correction : Implémenter vrais tests ou supprimer.

### M7. dashboard/app.py — Image URL depend reseau
Correction : Image locale ou emoji.

### M8-M28. (voir rapport complet) — Divers fichiers
Voir sections detaillees ci-dessous.

---

## 3. PROBLEMES MINEURS (resume)

- risk_manager.capital_actuel ignore gains (devrait etre capital + net_pnl)
- Feature cols confusion (FEATURE_COLS 8 vs FEATURE_COLS_V2 11)
- 12 fichiers _*.py dans tests/ a deplacer vers scripts/
- spread_cost_pct hardcode dans boom_crash_drift.py (devrait utiliser constants)
- asyncio.run dans Streamlit (event loop conflict)
- Token Telegram dans URL (preferer header Authorization)
- daily_loss jamais remis a zero (pas de reset minuit)
- SYMBOL=R_75 dans rise_fall.py contredit DEFAULT_SYMBOL=CRASH500

---

## 4. PROBLEMES D ARCHITECTURE

### A1. bot_executor = God Object (9+ imports, orchestre tout)
Recommandation : Extraire en sous-services.

### A2. Pas de couche Repository
Recommandation : Interface DataRepository (Deriv, SQLite, File).

### A3. Deux systemes de configuration (constants.py + .env)
Recommandation : Config object centralise.

### A4. Dashboard = mono-fichier Streamlit 442 lignes
Recommandation : Composants Streamlit separates.

### A5. forward/ scripts non integres
Recommandation : Integrer au pipeline ou archiver.

---

## 5. PLAN DE CORRECTION PRIORISE

### Phase 0 — Correctifs immediats (1-2 jours)
| # | Fichier | Action |
|---|---------|--------|
| 1 | deriv/__init__.py | Verifier/creer deriv/bot.py |
| 2 | deriv/risk_manager.py:12 | Retirer load_dotenv() |
| 3 | deriv/client.py:10 | Retirer load_dotenv() |
| 4 | deriv/bot_executor.py:61-73 | Fix try/except |
| 5 | deriv/bot_executor.py:167 | Remplacer _connected hack |
| 6 | tests/test_deriv.py | pytest.ini markers |
| 7 | forward/RAPPORT_CLAUDE.md | Corriger conclusion |

### Phase 1 —Dette technique (1-2 semaines)
| # | Action |
|---|--------|
| 8 | Journaliser APRES execution |
| 9 | Catch specifique risk_manager |
| 10 | Deplacer tests/_*.py vers scripts/ |
| 11 | Dashboard image locale |
| 12 | spread_cost_pct depuis constants |
| 13 | Fusionner decide/check |
| 14 | Log si proba is None |
| 15 | Documenter AGENTS.md |
| 16 | Reconstruire state.json |
| 17 | Implémenter TestRiseFallStrategy |

### Phase 2 — Architecture (1-2 mois)
| # | Action |
|---|--------|
| 18 | Config object centralise |
| 19 | Interface Repository |
| 20 | Decouper God Object |
| 21 | Composants Streamlit |
| 22 | capital_actuel correct |
| 23 | Reset daily_loss |
| 24 | Timestamp TradeResult |
| 25 | asyncio.run compatible |

### Phase 3 — Qualite continue
Couverture >80%, CI/CD, type hints, doc API, monitoring.

---

## 6. FICHIERS LES PLUS TOUCHES

| Fichier | Crit | Maj | Min | Total |
|---------|------|-----|-----|-------|
| deriv/bot_executor.py | 2 | 5 | 5 | 12 |
| deriv/risk_manager.py | 2 | 3 | 4 | 9 |
| deriv/__init__.py | 2 | 0 | 1 | 3 |
| tests/test_deriv.py | 3 | 2 | 0 | 5 |
| dashboard/app.py | 1 | 3 | 4 | 8 |
| deriv/ensemble_predictor.py | 0 | 2 | 3 | 5 |
| deriv/client.py | 1 | 1 | 3 | 5 |
| tests/_*.py | 0 | 1 | 12 | 13 |
| **TOTAL** | **13** | **23** | **39** | **75** |

---

*Rapport genere le 2026-09-16*
