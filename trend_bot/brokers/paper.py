"""Courtier papier : rejoue des clôtures (CSV ou DataFrame), ordres simulés, état en JSON.

Sert aux tests et à faire tourner le bot sans MetaTrader 5. Un lot = 1 unité de prix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trend_bot.brokers.base import OrderResult
from trend_bot.portfolio import SymbolSpec


class PaperBroker:
    def __init__(self, closes: pd.DataFrame, state_path: str | Path | None = None,
                 equity: float = 10_000.0, cost: float = 1e-4):
        self.closes = closes.sort_index()
        self.now = self.closes.index[-1]           # date « courante » (dernière clôture)
        self.cost = cost
        self.path = Path(state_path) if state_path else None
        self.cash, self.pos, self.entry = equity, {}, {}
        if self.path and self.path.exists():
            st = json.loads(self.path.read_text())
            self.cash, self.pos, self.entry = st["cash"], st["pos"], st["entry"]

    def _save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"cash": self.cash, "pos": self.pos,
                                             "entry": self.entry}))

    def _price(self, s: str) -> float:
        return float(self.closes[s][:self.now].dropna().iloc[-1])

    def resolve(self, logical: str, candidates: list[str]) -> str | None:
        for c in [logical, *candidates]:
            if c in self.closes.columns:
                return c
        return None

    def is_demo(self) -> bool:
        return True

    def equity(self) -> float:
        return self.cash + sum(l * (self._price(s) - self.entry[s]) for s, l in self.pos.items())

    def positions(self) -> dict[str, float]:
        return {s: l for s, l in self.pos.items() if l}

    def daily_closes(self, symbol: str, n: int) -> pd.Series:
        return self.closes[symbol][:self.now].dropna().iloc[-n:]

    def spec(self, symbol: str) -> SymbolSpec:
        return SymbolSpec(lot_value=self._price(symbol), volume_min=0.01,
                          volume_max=1e9, volume_step=0.01)

    def market_order(self, symbol: str, lots: float, comment: str = "") -> OrderResult:
        px = self._price(symbol)
        cur = self.pos.get(symbol, 0.0)
        if cur:                                   # réalise le PnL, repart du prix actuel
            self.cash += cur * (px - self.entry[symbol])
        self.cash -= abs(lots) * px * self.cost
        self.pos[symbol] = round(cur + lots, 8)
        self.entry[symbol] = px
        self._save()
        return OrderResult(True, lots, px, "papier")
