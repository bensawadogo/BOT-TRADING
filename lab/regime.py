"""Régime HMM CAUSAL : réentraîné sur le passé seulement, évalué bougie par bougie.

Reprend le modèle du projet (deriv.ensemble_predictor.HMMPredictor : 3 états,
features log-rendement + volatilité 20, labels par rendement moyen), mais
sans jamais voir les bougies futures et sans écrire de pickle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from deriv.ensemble_predictor import HMMPredictor, _hmm_features


class _HMM(HMMPredictor):
    def _save(self):   # pas d'écriture disque dans le labo
        pass

    def _load(self):
        pass


def causal_regimes(df: pd.DataFrame, train_len: int = 1500, retrain_every: int = 500,
                   window: int = 300) -> pd.DataFrame:
    """Pour chaque bougie i : régime et confiance calculés avec df[:i+1] seulement."""
    n = len(df)
    regime = np.array(["Range"] * n, dtype=object)
    conf = np.zeros(n)
    model = None
    for i in range(train_len, n):
        if model is None or (i - train_len) % retrain_every == 0:
            m = _HMM()
            try:
                m.train(df.iloc[i - train_len:i + 1])
                model = m
            except Exception:
                pass
        if model is None:
            continue
        feats = _hmm_features(df.iloc[max(0, i - window):i + 1])
        if len(feats) < 30:
            continue
        p = model.model.predict_proba(model.scaler.transform(feats.values))[-1]
        best = int(np.argmax(p))
        regime[i] = model.labels.get(best, "Range")
        conf[i] = float(p[best])
    return pd.DataFrame({"regime": regime, "confiance": conf}, index=df.index)
