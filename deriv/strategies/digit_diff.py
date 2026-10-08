"""Stratégie Digit Differs : évite un digit peu probable selon le bruit DWT."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from intelligence.signal_engine import SignalEngine, TradingSignal


class DigitDiffStrategy:
    def __init__(self, symbol: str = "R_100", barrier: str = "5", duration: int = 1):
        self.symbol = symbol
        self.barrier = barrier
        self.duration = duration
        self.engine = SignalEngine()

    def evaluate(self, prices: pd.Series) -> Optional[TradingSignal]:
        signal = self.engine.generate(prices)
        if signal.confidence < 0.4:
            return None
        return signal

    def to_proposal(self, signal: TradingSignal, stake: float) -> dict[str, Any]:
        return {
            "amount": stake,
            "basis": "stake",
            "contract_type": "DIGITDIFF",
            "currency": "USD",
            "duration": self.duration,
            "duration_unit": "t",
            "symbol": self.symbol,
            "barrier": self.barrier,
        }
