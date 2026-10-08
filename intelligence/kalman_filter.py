"""Filtre de Kalman pour estimer le vrai prix sous-jacent et éliminer le bruit."""

from __future__ import annotations

import numpy as np
import pandas as pd


class KalmanPriceFilter:
    """
    Filtre de Kalman pour estimer le "vrai prix" sous-jacent.
    Élimine le bruit des prix tick-by-tick.
    
    Signal :
      close > kalman_price * 1.01 → prix SURÉVALUÉ → potentiel SELL
      close < kalman_price * 0.99 → prix SOUS-ÉVALUÉ → potentiel BUY
    """

    def __init__(self, process_variance: float = 1e-5, observation_variance: float = 1e-3):
        self.Q = process_variance  # bruit du processus
        self.R = observation_variance  # bruit de mesure
        self.P = 1.0  # covariance initiale
        self.x: float | None = None  # estimation courante

    def update(self, price: float) -> float:
        """Met à jour le filtre avec un nouveau prix. Retourne le prix filtré."""
        if self.x is None:
            self.x = float(price)
            return self.x

        # Prédiction
        P_pred = self.P + self.Q

        # Mise à jour (gain de Kalman)
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (price - self.x)
        self.P = (1.0 - K) * P_pred

        return self.x

    def apply_to_series(self, prices: pd.Series) -> pd.Series:
        """Applique le filtre à une série de prix complète."""
        filtered = []
        self.x = None  # reset
        for p in prices:
            filtered.append(self.update(float(p)))
        return pd.Series(filtered, index=prices.index, name="kalman_price")

    def signal(self, current_price: float, threshold: float = 0.01) -> str:
        """
        Retourne le signal basé sur l'écart prix réel vs prix filtré.
        threshold = 1% d'écart minimum pour générer un signal.
        """
        if self.x is None:
            return "NEUTRAL"
        ecart = (current_price - self.x) / self.x
        if ecart > threshold:
            return "SELL"  # prix trop haut vs tendance
        elif ecart < -threshold:
            return "BUY"  # prix trop bas vs tendance
        return "NEUTRAL"

    # --- Méthodes de compatibilité ---
    def filter(self, prices: pd.Series) -> pd.DataFrame:
        """Retourne un DataFrame avec 'kalman_price' et 'kalman_velocity'."""
        k_series = self.apply_to_series(prices)
        velocity = k_series.diff().fillna(0.0)
        return pd.DataFrame(
            {"kalman_price": k_series, "kalman_velocity": velocity},
            index=prices.index,
        )

    def slope_sign(self, prices: pd.Series) -> int:
        """Signe de la pente pour compatibilité avec le signal engine."""
        df = self.filter(prices)
        v = float(df["kalman_velocity"].iloc[-1])
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0


# Alias pour rétrocompatibilité
KalmanFilter1D = KalmanPriceFilter
