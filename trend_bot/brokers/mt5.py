"""Courtier MetaTrader 5 (paquet officiel `MetaTrader5`, Windows uniquement).

Le terminal MT5 doit être installé et connecté à un compte (Deriv MT5 ou autre).
Identifiants facultatifs dans .env : MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH.
Le bot ne touche qu'à SES positions (numéro magique), en compte netting ou hedging.
"""
from __future__ import annotations

import logging
import math
import os

import pandas as pd

from trend_bot.brokers.base import OrderResult
from trend_bot.config import MAGIC
from trend_bot.portfolio import SymbolSpec

log = logging.getLogger(__name__)


class MT5Broker:
    def __init__(self, mt5_module=None):
        if mt5_module is None:
            import MetaTrader5 as mt5_module   # noqa: N813 — nom imposé par le paquet
        self.mt5 = mt5_module
        kw = {}
        if os.getenv("MT5_PATH"):
            kw["path"] = os.environ["MT5_PATH"]
        if os.getenv("MT5_LOGIN"):
            kw.update(login=int(os.environ["MT5_LOGIN"]),
                      password=os.getenv("MT5_PASSWORD", ""),
                      server=os.getenv("MT5_SERVER", ""))
        if not self.mt5.initialize(**kw):
            raise RuntimeError(f"MT5 initialize a échoué : {self.mt5.last_error()}")
        self._names = {s.name for s in (self.mt5.symbols_get() or [])}

    def close(self) -> None:
        self.mt5.shutdown()

    # ── lecture ──────────────────────────────────────────────────────────────
    def resolve(self, logical: str, candidates: list[str]) -> str | None:
        for c in [*candidates, logical]:
            if c in self._names:
                self.mt5.symbol_select(c, True)
                return c
        return None

    def is_demo(self) -> bool | None:
        acc = self.mt5.account_info()
        if acc is None:
            return None
        return acc.trade_mode == self.mt5.ACCOUNT_TRADE_MODE_DEMO

    def equity(self) -> float:
        return float(self.mt5.account_info().equity)

    def _own(self, symbol: str | None = None) -> list:
        pos = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [p for p in (pos or []) if p.magic == MAGIC]

    def positions(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self._own():
            sgn = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
            out[p.symbol] = round(out.get(p.symbol, 0.0) + sgn * p.volume, 8)
        return {s: v for s, v in out.items() if v}

    def daily_closes(self, symbol: str, n: int) -> pd.Series:
        # position 1 = dernière bougie TERMINÉE (0 = bougie du jour en cours)
        rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_D1, 1, n)
        if rates is None or len(rates) == 0:
            return pd.Series(dtype=float)
        df = pd.DataFrame(rates)
        idx = pd.to_datetime(df["time"], unit="s").dt.normalize()
        return pd.Series(df["close"].to_numpy(float), index=idx, name=symbol)

    def spec(self, symbol: str) -> SymbolSpec:
        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)
        price = (tick.bid + tick.ask) / 2
        # valeur d'1 lot en devise du compte = prix × (valeur d'un tick / taille d'un tick)
        lot_value = price * info.trade_tick_value / info.trade_tick_size
        return SymbolSpec(lot_value, info.volume_min, info.volume_max, info.volume_step)

    # ── ordres ───────────────────────────────────────────────────────────────
    def _filling(self, symbol: str) -> int:
        fm = self.mt5.symbol_info(symbol).filling_mode
        if fm & 1:
            return self.mt5.ORDER_FILLING_FOK
        if fm & 2:
            return self.mt5.ORDER_FILLING_IOC
        return self.mt5.ORDER_FILLING_RETURN

    def _deal(self, symbol: str, lots: float, comment: str, ticket: int | None = None):
        tick = self.mt5.symbol_info_tick(symbol)
        buy = lots > 0
        req = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(round(abs(lots), 8)),
            "type": self.mt5.ORDER_TYPE_BUY if buy else self.mt5.ORDER_TYPE_SELL,
            "price": tick.ask if buy else tick.bid,
            "deviation": 20,
            "magic": MAGIC,
            "comment": comment[:31],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self._filling(symbol),
        }
        if ticket is not None:
            req["position"] = int(ticket)
        res = self.mt5.order_send(req)
        ok = res is not None and res.retcode in (self.mt5.TRADE_RETCODE_DONE,
                                                 self.mt5.TRADE_RETCODE_PLACED)
        msg = f"retcode={getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}".strip()
        if not ok:
            log.error("ordre refusé %s %.2f : %s / %s", symbol, lots, msg, self.mt5.last_error())
        return ok, (float(res.volume) if ok else 0.0), (float(res.price) if ok else None), msg

    def market_order(self, symbol: str, lots: float, comment: str = "trend_bot") -> OrderResult:
        step = self.mt5.symbol_info(symbol).volume_step
        reste = math.copysign(round(abs(lots) / step) * step, lots)
        fait, prix, msgs = 0.0, None, []
        hedging = self.mt5.account_info().margin_mode == self.mt5.ACCOUNT_MARGIN_MODE_RETAIL_HEDGING
        if hedging:   # réduire d'abord les positions opposées, ticket par ticket
            for p in self._own(symbol):
                sgn = 1 if p.type == self.mt5.POSITION_TYPE_BUY else -1
                if sgn * reste >= 0 or abs(reste) < step / 2:
                    continue
                vol = min(p.volume, abs(reste))
                ok, v, px, m = self._deal(symbol, math.copysign(vol, reste), comment, p.ticket)
                msgs.append(m)
                if not ok:
                    return OrderResult(False, fait, prix, "; ".join(msgs))
                fait += math.copysign(v, reste)
                reste -= math.copysign(v, reste)
                prix = px
        if abs(reste) >= step / 2:
            ok, v, px, m = self._deal(symbol, reste, comment)
            msgs.append(m)
            if not ok:
                return OrderResult(False, fait, prix, "; ".join(msgs))
            fait += math.copysign(v, reste)
            prix = px
        return OrderResult(True, round(fait, 8), prix, "; ".join(msgs))
