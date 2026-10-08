"""Analyse Discrete Wavelet Transform pour extraire tendance et bruit."""

from __future__ import annotations

import numpy as np
import pandas as pd


class DWTAnalyzer:
    def __init__(self, wavelet: str = "db4", level: int = 3):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, prices: pd.Series) -> pd.DataFrame:
        try:
            import pywt
        except ImportError as exc:
            raise ImportError("PyWavelets (pywt) est requis pour DWT.") from exc

        values = prices.astype(float).to_numpy()
        coeffs = pywt.wavedec(values, self.wavelet, level=self.level)
        trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        noise_coeffs = [np.zeros_like(coeffs[0])] + list(coeffs[1:])
        trend = pywt.waverec(trend_coeffs, self.wavelet)[: len(values)]
        noise = pywt.waverec(noise_coeffs, self.wavelet)[: len(values)]
        return pd.DataFrame(
            {"dwt_trend": trend, "dwt_noise": noise},
            index=prices.index,
        )

    def trend_direction(self, prices: pd.Series, lookback: int = 8) -> int:
        d = self.decompose(prices)
        trend = d["dwt_trend"]
        if len(trend) < lookback + 1:
            return 0
        delta = float(trend.iloc[-1] - trend.iloc[-lookback])
        if delta > 0:
            return 1
        if delta < 0:
            return -1
        return 0
