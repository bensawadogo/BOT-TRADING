"""Détection de régime de marché via Hidden Markov Model (Bull / Bear / Range)."""

from __future__ import annotations

import logging
import os
import pickle
from deriv.constants import safe_pickle_load
from enum import IntEnum
from typing import Optional, Union

from hmmlearn import hmm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Regime(IntEnum):
    BEAR = 0
    RANGE = 1
    BULL = 2


class HMMRegimeDetector:
    """
    Détecte le régime de marché via HMM Gaussien.
    États : 0=Bull (haussier) · 1=Bear (baissier) · 2=Range (latéral)
    
    Usage :
        detector = HMMRegimeDetector()
        detector.train(df_ohlcv)            # 1 fois
        regime = detector.predict(df_ohlcv) # en continu
    """

    N_STATES = 3
    MODEL_PATH = "intelligence/models/hmm_model.pkl"

    def __init__(self):
        self.model = hmm.GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="diag",
            n_iter=200,
            random_state=42,
        )
        self.is_trained = False
        self.regime_labels = {}  # {0: "Bull", 1: "Bear", 2: "Range"}

    def _resolve_model_path(self) -> str:
        """Résout le chemin du modèle de manière robuste."""
        if os.path.exists(self.MODEL_PATH):
            return self.MODEL_PATH
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidate = os.path.join(base_dir, self.MODEL_PATH)
        if os.path.exists(candidate):
            return candidate
        return self.MODEL_PATH

    def prepare_features(self, df: pd.DataFrame | pd.Series) -> np.ndarray:
        """
        Features pour le HMM :
        - Rendement log (direction)
        - Volatilité 20 périodes (stabilité)
        - Volume relatif (confirmation)
        """
        if isinstance(df, pd.Series):
            df = pd.DataFrame({"close": df, "volume": 1.0})
        else:
            df = df.copy()

        if "volume" not in df.columns:
            df["volume"] = 1.0
        else:
            df["volume"] = df["volume"].fillna(1.0)
            if (df["volume"] == 0).all():
                df["volume"] = 1.0

        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["volatility"] = df["log_return"].rolling(20).std()

        vol_roll = df["volume"].rolling(20).mean().replace(0, np.nan)
        df["vol_relative"] = (df["volume"] / vol_roll).fillna(1.0)

        df = df.dropna(subset=["log_return", "volatility"])
        return df[["log_return", "volatility", "vol_relative"]].values

    def train(self, df: pd.DataFrame | pd.Series) -> dict:
        """
        Entraîne le HMM sur l'historique OHLCV.
        Nécessite au moins 200 bougies.
        """
        if len(df) < 200:
            raise ValueError(f"Besoin de 200+ bougies, reçu {len(df)}")

        features = self.prepare_features(df)
        self.model.fit(features)
        self.is_trained = True

        # Labelliser les états automatiquement
        states = self.model.predict(features)
        returns_by_state = {}
        for s in range(self.N_STATES):
            mask = states == s
            mean_return = features[mask, 0].mean() if mask.sum() > 0 else 0
            mean_vol = features[mask, 1].mean() if mask.sum() > 0 else 0
            returns_by_state[s] = (mean_return, mean_vol)

        # État avec return le plus bas = Bear
        # État au milieu = Range
        # État avec return le plus haut = Bull
        sorted_states = sorted(returns_by_state.items(), key=lambda x: x[1][0])
        self.regime_labels = {
            sorted_states[0][0]: "Bear",
            sorted_states[1][0]: "Range",
            sorted_states[2][0]: "Bull",
        }

        # Sauvegarde
        save_path = self._resolve_model_path()
        model_dir = os.path.dirname(save_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"model": self.model, "labels": self.regime_labels}, f)

        # Matrice de transition
        logger.info(f"HMM entraîné. Labels: {self.regime_labels}")
        logger.info(f"Matrice transition:\n{self.model.transmat_}")

        return {
            "regime_labels": self.regime_labels,
            "transition_matrix": self.model.transmat_.tolist(),
            "n_samples": len(df),
        }

    def fit(self, df_or_prices: pd.DataFrame | pd.Series) -> "HMMRegimeDetector":
        """Alias de compatibilité avec les interfaces scikit-learn / stratégies existantes."""
        self.train(df_or_prices)
        return self

    def _ensure_trained(self) -> None:
        if not self.is_trained:
            resolved_path = self._resolve_model_path()
            if os.path.exists(resolved_path):
                with open(resolved_path, "rb") as f:
                    data = safe_pickle_load(f)
                    self.model = data["model"]
                    self.regime_labels = data["labels"]
                    self.is_trained = True
            else:
                raise RuntimeError("Modèle non entraîné. Lance train() d'abord.")

    def predict(self, df: pd.DataFrame | pd.Series) -> dict | pd.Series:
        """
        Prédit le régime actuel + probabilités.
        Retourne le régime de la DERNIÈRE bougie (dict).
        Pour une pd.Series, retourne la série complète des régimes pour rétrocompatibilité.
        """
        self._ensure_trained()
        features = self.prepare_features(df)
        states = self.model.predict(features)
        probas = self.model.predict_proba(features)

        if isinstance(df, pd.Series):
            label_to_regime = {"Bear": Regime.BEAR, "Range": Regime.RANGE, "Bull": Regime.BULL}
            mapped = [int(label_to_regime.get(self.regime_labels[s], Regime.RANGE)) for s in states]
            idx = df.index[-len(mapped):]
            return pd.Series(mapped, index=idx, name="regime")

        current_state = states[-1]
        current_probas = probas[-1]

        label_probas = {
            self.regime_labels[s]: round(float(current_probas[s]), 3)
            for s in range(self.N_STATES)
        }
        p_bull = label_probas.get("Bull", 0.0)
        p_bear = label_probas.get("Bear", 0.0)
        p_range = label_probas.get("Range", 0.0)

        return {
            "regime": self.regime_labels[current_state],
            "régime": self.regime_labels[current_state],
            "state_id": int(current_state),
            "confiance": round(float(current_probas[current_state]), 3),
            "P_Bull": p_bull,
            "P_Bear": p_bear,
            "P_Range": p_range,
            "probas": label_probas,
        }


    def current_regime(self, df_or_prices: pd.DataFrame | pd.Series) -> Regime:
        """Retourne le régime courant sous forme d'énumération Regime."""
        pred = self.predict(df_or_prices)
        if isinstance(pred, pd.Series):
            val = int(pred.iloc[-1]) if len(pred) else int(Regime.RANGE)
            return Regime(val)
        label = pred["regime"]
        return {"Bull": Regime.BULL, "Bear": Regime.BEAR, "Range": Regime.RANGE}.get(label, Regime.RANGE)

