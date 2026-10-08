"""Configuration pytest globale.

Les tests ne doivent JAMAIS écraser les modèles entraînés de production
(deriv/models/*.pkl, intelligence/models/*.pkl) : chaque test écrit dans
un répertoire temporaire isolé.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _isole_modeles_production(tmp_path, monkeypatch):
    """Redirige toutes les sauvegardes de modèles vers un dossier temporaire."""
    _model_paths = [
        ("deriv.ensemble_predictor", "MODELS_DIR"),
        ("deriv.online_learner", "MODELS_DIR"),
        ("deriv.online_learner", "STATE_FILE"),
        ("deriv.weekly_retrain", "DEFAULT_STATE_FILE"),
        ("intelligence.hmm_regime", "MODEL_PATH"),
    ]
    for module_name, attr_name in _model_paths:
        try:
            mod = __import__(module_name, fromlist=[attr_name])
            monkeypatch.setattr(mod, attr_name, str(tmp_path / "models"))
        except Exception:
            pass
    yield

