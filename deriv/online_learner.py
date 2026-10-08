"""
════════════════════════════════════════════════════════════════════
ONLINE LEARNING — adaptation en temps réel au concept drift
deriv/online_learner.py
════════════════════════════════════════════════════════════════════

Le modèle batch (HMM/XGBoost) est ré-entraîné 1×/semaine. Or le marché
change de régime plus vite (KDD 2025 — Proactive Adaptation Against
Drift). Ce module complète le batch avec un modèle ONLINE qui :

  1. apprend après CHAQUE trade fermé (boucle fermée journal → modèle),
  2. détecte le concept drift avec ADWIN (adaptive windowing),
  3. expose un 8e vote pour l'ensemble (0 / 0.5 / 1).

Modèle : régression logistique online (River) + StandardScaler.
Contrairement au HoeffdingAdaptiveTree (borne de Hoeffding : aucun split
avant plusieurs centaines d'échantillons — vérifié : n_leaves=1 après
200 exemples séparables), la régression logistique apprend dès le
premier exemple et convient au régime n < 1000 trades du journal.
Le drift reste détecté par ADWIN sur les erreurs du modèle.

⚠️ Le vote online est INFO par défaut : la règle 4/5 reste verrouillée
sur les 7 modèles batch (MIN_VOTES_TO_TRADE inchangé).
"""
from __future__ import annotations

import logging
import os
import pickle
from deriv.constants import safe_pickle_load

logger = logging.getLogger(__name__)

try:
    from river import drift, linear_model, metrics, optim, preprocessing
    _RIVER_OK = True
    _RIVER_ERROR = ""
except Exception as exc:  # pragma: no cover
    _RIVER_OK = False
    _RIVER_ERROR = repr(exc)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_BASE_DIR, "models")
STATE_FILE = os.path.join(MODELS_DIR, "online_learner.pkl")

# Features du modèle online (disponibles dans JournalEntry à l'entrée)
FEATURES_ONLINE = [
    "hmm_confidence", "ensemble_votes", "rsi", "adx",
    "volume_ratio", "atr", "risk_reward",
    "session_london", "session_overlap", "session_newyork",
    "hour_sin", "hour_cos", "day_of_week",
]


class OnlineLearner:
    """Apprentissage en ligne après chaque trade fermé (River)."""

    def __init__(self, state_file: str | None = None):
        self.state_file = state_file or STATE_FILE
        self.n_samples = 0
        self.n_drifts = 0
        if _RIVER_OK:
            self.model = (
                preprocessing.StandardScaler()
                | linear_model.LogisticRegression(
                    optimizer=optim.SGD(0.05),
                    l2=1e-4,
                )
            )
            self.drift_detector = drift.ADWIN(delta=0.002)
            self.accuracy = metrics.Accuracy()
        else:
            self.model = None
            self.drift_detector = None
            self.accuracy = None
            logger.warning(
                "River indisponible : online learner désactivé (%s)",
                _RIVER_ERROR,
            )
        self._load()

    # ── Apprentissage ──────────────────────────────────────────────
    def update(self, features: dict, label: int) -> dict:
        """
        Met à jour le modèle avec UN trade fermé.

        features : dict au moment de l'entrée (cf. FEATURES_ONLINE)
        label    : 1 si le trade a gagné, 0 sinon
        """
        if not _RIVER_OK or self.model is None:
            return {"ok": False, "reason": "river indisponible"}

        x = {k: float(features.get(k, 0.0) or 0.0) for k in FEATURES_ONLINE}
        y = int(label)

        # Prévision AVANT apprentissage (mesure honnête de l'accuracy)
        pred = self.model.predict_one(x)
        if pred is None:
            logger.warning("predict_one retourne None")
        else:
            self.accuracy.update(y, pred)
            self.drift_detector.update(1 if pred != y else 0)

        self.model.learn_one(x, y)
        self.n_samples += 1

        drift_detected = False
        if self.drift_detector.drift_detected:
            self.n_drifts += 1
            drift_detected = True
            logger.warning(
                "CONCEPT DRIFT #%d détecté après %d trades | "
                "accuracy online courante : %.1f%%",
                self.n_drifts, self.n_samples, self.accuracy.get() * 100,
            )
            # La régression logistique s'adapte en continu via SGD —
            # pas de reset manuel (adaptation proactive).

        self._save()
        return {
            "ok": True,
            "accuracy_running": round(float(self.accuracy.get()), 3),
            "n_samples": self.n_samples,
            "n_drifts": self.n_drifts,
            "drift_detected": drift_detected,
        }

    # ── Prédiction ─────────────────────────────────────────────────
    def predict(self, features: dict) -> dict:
        """Vote online (0 / 0.5 / 1) — neutre si < 30 trades appris."""
        if (not _RIVER_OK or self.model is None
                or self.n_samples < 30):
            return {"vote": 0.5, "proba_up": 0.5, "n_trained": self.n_samples,
                    "active": False}
        x = {k: float(features.get(k, 0.0) or 0.0) for k in FEATURES_ONLINE}
        proba = self.model.predict_proba_one(x)
        p_up = float((proba or {}).get(1, 0.5))
        vote = 1 if p_up > 0.58 else 0 if p_up < 0.42 else 0.5
        return {"vote": vote, "proba_up": round(p_up, 3),
                "n_trained": self.n_samples, "active": True}

    # ── Persistance ────────────────────────────────────────────────
    def _save(self) -> None:
        if not _RIVER_OK:
            return
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(self.state_file, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "drift": self.drift_detector,
                    "accuracy": self.accuracy,
                    "n_samples": self.n_samples,
                    "n_drifts": self.n_drifts,
                }, f)
        except Exception:
            logger.warning("Sauvegarde online learner impossible",
                           exc_info=True)

    def _load(self) -> None:
        if not _RIVER_OK or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, "rb") as f:
                data = safe_pickle_load(f)
            self.model = data.get("model", self.model)
            self.drift_detector = data.get("drift", self.drift_detector)
            self.accuracy = data.get("accuracy", self.accuracy)
            self.n_samples = int(data.get("n_samples", 0))
            self.n_drifts = int(data.get("n_drifts", 0))
            logger.info("Online learner chargé : %d samples, %d drifts",
                        self.n_samples, self.n_drifts)
        except Exception:
            logger.warning("Chargement online learner impossible — reset",
                           exc_info=True)



