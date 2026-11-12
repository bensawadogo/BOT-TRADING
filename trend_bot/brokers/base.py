from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from trend_bot.portfolio import SymbolSpec


@dataclass
class OrderResult:
    ok: bool
    lots: float            # lots exécutés (signés)
    price: float | None
    message: str = ""


class Broker(Protocol):
    def resolve(self, logical: str, candidates: list[str]) -> str | None:
        """Nom réel du marché chez le courtier, ou None s'il n'existe pas."""

    def is_demo(self) -> bool | None:
        """True compte démo, False compte réel, None inconnu."""

    def equity(self) -> float: ...

    def positions(self) -> dict[str, float]:
        """Lots nets signés par marché (positions du bot seulement)."""

    def daily_closes(self, symbol: str, n: int) -> pd.Series:
        """Clôtures journalières TERMINÉES (sans la bougie en cours), index date."""

    def spec(self, symbol: str) -> SymbolSpec: ...

    def market_order(self, symbol: str, lots: float, comment: str) -> OrderResult:
        """Achète (lots > 0) ou vend (lots < 0) au marché."""
