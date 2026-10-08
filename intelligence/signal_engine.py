"""Moteur de signal fusionnant HMM, Filtre de Kalman et indicateurs techniques."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import ta

from .dwt_analysis import DWTAnalyzer
from .hmm_regime import HMMRegimeDetector, Regime
from .kalman_filter import KalmanPriceFilter, KalmanFilter1D

logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    direction: int  # -1 short / 0 neutre / +1 long
    confidence: float
    regime: Regime
    kalman_slope: int
    dwt_trend: int
    reason: str
    score: float = 50.0
    signal_str: str = "HOLD"
    details: Optional[dict[str, Any]] = None


class SignalEngine:
    """
    Fusionne HMM + Kalman + indicateurs techniques → signal final.
    Score de 0 à 100 : > 65 = BUY, < 35 = SELL, entre = HOLD
    """

    def __init__(self):
        self.hmm = HMMRegimeDetector()
        self.kalman = KalmanPriceFilter()
        self.dwt = DWTAnalyzer()
        self._fitted = False

    def calculate_score(self, df: pd.DataFrame) -> dict:
        """
        Calcule le score de trading sur les données OHLCV.
        df doit avoir les colonnes : open, high, low, close, volume.
        Retourne un dict avec score, signal, détails.
        """
        if isinstance(df, pd.Series):
            df = pd.DataFrame({
                "open": df,
                "high": df,
                "low": df,
                "close": df,
                "volume": 1.0,
            })
        else:
            df = df.copy()

        # Sécurisation des colonnes pour compatibilité Deriv / crypto partiel
        if "close" not in df.columns and len(df.columns) > 0:
            df["close"] = df.iloc[:, 0]
        if "volume" not in df.columns:
            df["volume"] = 1.0
        if "open" not in df.columns:
            df["open"] = df["close"]
        if "high" not in df.columns:
            df["high"] = df["close"]
        if "low" not in df.columns:
            df["low"] = df["close"]

        if len(df) < 50:
            return {"score": 50, "signal": "HOLD", "raison": "Pas assez de données", "details": {}, "force": "FAIBLE"}

        score = 50.0  # neutre par défaut
        details: dict[str, Any] = {}

        # 1. RÉGIME HMM (poids 40%)
        try:
            regime_data = self.hmm.predict(df)
            regime = regime_data["regime"]
            confiance = regime_data["confiance"]

            if regime == "Bull" and confiance > 0.7:
                score += 20
            elif regime == "Bear" and confiance > 0.7:
                score -= 20
            # Range = neutre, on ne bouge pas
            details["regime"] = f"{regime} ({confiance:.0%})"
        except Exception:
            details["regime"] = "Non disponible"

        # 2. KALMAN FILTER (poids 20%)
        try:
            self.kalman.apply_to_series(df["close"])
            kalman_signal = self.kalman.signal(float(df["close"].iloc[-1]))
            if kalman_signal == "BUY":
                score += 10
            elif kalman_signal == "SELL":
                score -= 10
            details["kalman"] = kalman_signal
        except Exception:
            details["kalman"] = "NEUTRAL"

        # 3. RSI (poids 15%)
        try:
            rsi = ta.momentum.RSIIndicator(df["close"], window=14).rsi().iloc[-1]
            if rsi < 35:
                score += 10
            elif rsi > 65:
                score -= 10
            details["rsi"] = round(float(rsi), 1)
        except Exception:
            details["rsi"] = 50.0

        # 4. MACD (poids 15%)
        try:
            macd_obj = ta.trend.MACD(df["close"])
            macd_val = macd_obj.macd().iloc[-1]
            macd_sig = macd_obj.macd_signal().iloc[-1]
            if macd_val > macd_sig:
                score += 7
            else:
                score -= 7
            details["macd"] = "HAUSSIER" if macd_val > macd_sig else "BAISSIER"
        except Exception:
            details["macd"] = "INCONNU"

        # 5. BOLLINGER BANDS (poids 10%)
        try:
            bb = ta.volatility.BollingerBands(df["close"], window=20)
            close = float(df["close"].iloc[-1])
            bb_low = float(bb.bollinger_lband().iloc[-1])
            bb_high = float(bb.bollinger_hband().iloc[-1])
            if close < bb_low:
                score += 5  # sous la bande basse = potentiel rebond
            elif close > bb_high:
                score -= 5
            details["bollinger"] = (
                "SOUS BANDE" if close < bb_low else "SUR BANDE" if close > bb_high else "DANS BANDE"
            )
        except Exception:
            details["bollinger"] = "DANS BANDE"

        # Score final entre 0 et 100
        score = max(0.0, min(100.0, float(score)))

        signal = "BUY" if score > 65 else "SELL" if score < 35 else "HOLD"

        return {
            "score": round(score, 1),
            "signal": signal,
            "details": details,
            "force": "FORT" if abs(score - 50) > 25 else "MOYEN" if abs(score - 50) > 10 else "FAIBLE",
        }

    def decide(self, df: pd.DataFrame) -> dict:
        """
        Couche 3 — Logique de décision du bot :
          Condition BUY : HMM Bull > 0.70 ET Kalman BUY ET consensus > 65
          Condition SELL: HMM Bear > 0.70 ET Kalman SELL ET consensus < 35
          Sinon -> HOLD
        """
        # Couche 1 — HMM
        try:
            hmm_out = self.hmm.predict(df)
        except Exception:
            hmm_out = {
                "régime": "Range",
                "regime": "Range",
                "confiance": 0.5,
                "P_Bull": 0.33,
                "P_Bear": 0.33,
                "P_Range": 0.34,
            }

        p_bull = hmm_out.get("P_Bull", 0.0)
        p_bear = hmm_out.get("P_Bear", 0.0)

        # Couche 2 — Filtres
        analysis = self.calculate_score(df)
        consensus_score = analysis["score"]
        kalman_sig = analysis["details"].get("kalman", "NEUTRAL")

        series_close = df["close"] if "close" in df.columns else df.iloc[:, 0]
        try:
            dwt_sign = self.dwt.trend_direction(series_close)
        except Exception:
            dwt_sign = 0
        dwt_trend = "UP" if dwt_sign > 0 else "DOWN" if dwt_sign < 0 else "NEUTRAL"

        filters_out = {
            "kalman_signal": kalman_sig,
            "dwt_trend": dwt_trend,
            "consensus_score": consensus_score,
        }

        # Couche 3 — Décision
        if p_bull > 0.70 and kalman_sig == "BUY" and consensus_score > 65:
            action = "BUY"
            reason = f"Triple confirmation BUY: HMM Bull ({p_bull:.0%}) + Kalman BUY + Consensus {consensus_score}/100"
        elif p_bear > 0.70 and kalman_sig == "SELL" and consensus_score < 35:
            action = "SELL"
            reason = f"Triple confirmation SELL: HMM Bear ({p_bear:.0%}) + Kalman SELL + Consensus {consensus_score}/100"
        else:
            action = "HOLD"
            reason = "Conditions de triple convergence non satisfaites -> HOLD"


        return {
            "action": action,
            "reason": reason,
            "hmm": hmm_out,
            "filters": filters_out,
        }

    def message_telegram(self, pair: str, analyse: dict) -> str:

        """Génère le message d'alerte Telegram formaté avec emojis."""
        emoji = "🟢" if analyse.get("signal") == "BUY" else "🔴" if analyse.get("signal") == "SELL" else "⚪"
        d = analyse.get("details", {})
        force = analyse.get("force", "MOYEN")
        score = analyse.get("score", 50.0)
        signal = analyse.get("signal", "HOLD")

        return (
            f"{emoji} *{pair}* — Signal *{signal}* ({force})\n"
            f"Score : {score}/100\n\n"
            f"📊 Régime HMM : {d.get('regime', '?')}\n"
            f"📈 Kalman : {d.get('kalman', '?')}\n"
            f"💹 RSI : {d.get('rsi', '?')}\n"
            f"📉 MACD : {d.get('macd', '?')}\n"
            f"🎯 Bollinger : {d.get('bollinger', '?')}"
        )

    # --- Passerelles de rétrocompatibilité ---
    def fit(self, prices_or_df: pd.Series | pd.DataFrame) -> "SignalEngine":
        """Entraîne le composant HMM interne sur la série ou le DataFrame."""
        self.hmm.fit(prices_or_df)
        self._fitted = True
        return self

    def generate(self, prices: pd.Series | pd.DataFrame) -> TradingSignal:
        """Produit un TradingSignal complet compatible avec l'écosystème Deriv et Streamlit."""
        if isinstance(prices, pd.Series):
            df = pd.DataFrame({
                "open": prices,
                "high": prices,
                "low": prices,
                "close": prices,
                "volume": 1.0,
            })
            series_prices = prices
        else:
            df = prices.copy()
            series_prices = df["close"] if "close" in df.columns else df.iloc[:, 0]

        analysis = self.calculate_score(df)
        sig = analysis["signal"]
        direction = 1 if sig == "BUY" else -1 if sig == "SELL" else 0
        confidence = abs(analysis["score"] - 50.0) / 50.0

        try:
            regime = self.hmm.current_regime(df)
        except Exception:
            regime = Regime.RANGE

        k_slope = self.kalman.slope_sign(series_prices)
        dwt_dir = self.dwt.trend_direction(series_prices)

        reason = (
            f"score={analysis['score']} signal={sig} ({analysis['force']}) "
            f"regime={regime.name} kalman={analysis['details'].get('kalman', '?')}"
        )

        return TradingSignal(
            direction=direction,
            confidence=float(confidence),
            regime=regime,
            kalman_slope=k_slope,
            dwt_trend=dwt_dir,
            reason=reason,
            score=analysis["score"],
            signal_str=sig,
            details=analysis["details"],
        )
