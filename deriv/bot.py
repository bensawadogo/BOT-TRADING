"""Orchestrateur Deriv : connecte le client, le risk manager et les stratégies."""

from __future__ import annotations

import asyncio
from typing import Optional

import pandas as pd

from .client import DerivClient
from .risk_manager import RiskManager
from .strategies.digit_diff import DigitDiffStrategy
from .strategies.over_under import OverUnderStrategy
from .strategies.rise_fall import RiseFallStrategy


class DerivBot:
    def __init__(self, strategy_name: str = "rise_fall", stake: float = 0.5):
        self.client = DerivClient()
        self.risk = RiskManager()
        self.stake = stake
        self.strategy = {
            "rise_fall": RiseFallStrategy(),
            "over_under": OverUnderStrategy(),
            "digit_diff": DigitDiffStrategy(),
        }[strategy_name]

    async def start(self, prices: Optional[pd.Series] = None, dry_run: bool = True) -> None:
        await self.client.connect()
        try:
            if prices is None or prices.empty:
                print("DerivBot connecté (pas de série de prix : idle).")
                await self.client.ping()
                return
            signal = self.strategy.evaluate(prices)
            if signal is None:
                print("Aucun signal Deriv.")
                return
            decision = self.risk.decide(self.stake)
            print(f"Signal {signal.reason} | {decision.message}")
            if not decision.allowed:
                return
            proposal = self.strategy.to_proposal(signal, decision.stake)
            if dry_run:
                print("DRY-RUN proposal:", proposal)
                return
            await self.client.proposal(proposal)
        finally:
            await self.client.close()


if __name__ == "__main__":
    asyncio.run(DerivBot().start(dry_run=True))
