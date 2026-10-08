"""Stratégie Over/Under (dernier digit) guidée par le régime HMM."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from intelligence.hmm_regime import Regime
from intelligence.signal_engine import SignalEngine, TradingSignal


class OverUnderStrategy:
    def __init__(self, symbol: str = "R_100", barrier: str = "5", duration: int = 1):
        self.symbol = symbol
        self.barrier = barrier
        self.duration = duration
        self.engine = SignalEngine()

    def evaluate(self, prices: pd.Series) -> Optional[TradingSignal]:
        signal = self.engine.generate(prices)
        if signal.regime == Regime.RANGE and signal.confidence < 0.6:
            return None
        return signal

    def to_proposal(self, signal: TradingSignal, stake: float) -> dict[str, Any]:
        contract = "DIGITOVER" if signal.direction >= 0 else "DIGITUNDER"
        return {
            "amount": stake,
            "basis": "stake",
            "contract_type": contract,
            "currency": "USD",
            "duration": self.duration,
            "duration_unit": "t",
            "symbol": self.symbol,
            "barrier": self.barrier,
        }
