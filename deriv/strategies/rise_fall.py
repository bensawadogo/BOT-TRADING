"""Stratégie Rise/Fall pour indices synthétiques Deriv."""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import pandas as pd

# Résolution robuste de la racine du projet
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from intelligence.signal_engine import SignalEngine, TradingSignal


class RiseFallStrategy:
    """
    Stratégie Rise/Fall sur indices synthétiques Deriv.
    Utilise le SignalEngine (HMM + Kalman + indicateurs).
    
    Marchés recommandés :
      R_75  = Volatility 75 Index  (le plus populaire)
      R_100 = Volatility 100 Index (plus volatil)
      R_25  = Volatility 25 Index  (plus calme)
    """

    SYMBOL = "R_75"  # index par défaut
    DURATION = 60  # 1 minute
    STAKE_AUTO = 2.0  # 2$ en auto (≈ 1200 XOF)
    MIN_SCORE = 68  # score minimum pour trader

    contract_type_map = {1: "CALL", -1: "PUT"}

    def __init__(self, symbol: str = SYMBOL, duration: int = DURATION, duration_unit: str = "s"):
        self.symbol = symbol
        self.duration = duration
        self.duration_unit = duration_unit
        self.engine = SignalEngine()

    def should_trade(self, candles: list | pd.DataFrame) -> dict:
        """
        Analyse les bougies et décide si on trade.
        candles : liste de dicts {open, high, low, close, volume} ou objets bougies.
        """
        if isinstance(candles, pd.DataFrame):
            df = candles.copy()
        else:
            first = candles[0] if candles else {}
            if hasattr(first, "model_dump"):
                df = pd.DataFrame([c.model_dump() for c in candles])
            elif hasattr(first, "__dict__"):
                df = pd.DataFrame([c.__dict__ for c in candles])
            else:
                df = pd.DataFrame(candles)

        if len(df.columns) == 5 and all(isinstance(col, int) for col in df.columns):
            df.columns = ["open", "high", "low", "close", "volume"]
        elif "close" in df.columns:
            for col in ["open", "high", "low"]:
                if col not in df.columns:
                    df[col] = df["close"]
            if "volume" not in df.columns:
                df["volume"] = 1.0
            df = df[["open", "high", "low", "close", "volume"]]
        elif len(df.columns) == 5:
            df.columns = ["open", "high", "low", "close", "volume"]

        df = df.astype(float)

        analyse = self.engine.calculate_score(df)

        if analyse["score"] >= self.MIN_SCORE:
            return {
                "trade": True,
                "contract_type": "CALL",  # Rise
                "montant": self.STAKE_AUTO,
                "raison": analyse,
            }
        elif analyse["score"] <= (100 - self.MIN_SCORE):
            return {
                "trade": True,
                "contract_type": "PUT",  # Fall
                "montant": self.STAKE_AUTO,
                "raison": analyse,
            }
        else:
            return {"trade": False, "raison": analyse}

    # --- Passerelles de compatibilité pour DerivBot existant ---
    def evaluate(self, prices: pd.Series) -> Optional[TradingSignal]:
        signal = self.engine.generate(prices)
        if signal.direction == 0 or signal.confidence < 0.5:
            return None
        return signal

    def to_proposal(self, signal: TradingSignal, stake: float) -> dict[str, Any]:
        return {
            "amount": stake,
            "basis": "stake",
            "contract_type": self.contract_type_map.get(signal.direction, "CALL"),
            "currency": "USD",
            "duration": self.duration,
            "duration_unit": self.duration_unit,
            "symbol": self.symbol,
        }
