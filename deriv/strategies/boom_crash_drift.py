"""
════════════════════════════════════════════════════════════════════
STRATÉGIE BOOM/CRASH DRIFT — Exploite le drift structurel
deriv/strategies/boom_crash_drift.py
════════════════════════════════════════════════════════════════════

INSIGHT CLÉ (vérifié académiquement) :
  Boom 500 : entre deux spikes, le prix DESCEND lentement (drift baissier)
  Crash 500 : entre deux spikes, le prix MONTE lentement (drift haussier)

Les spikes sont aléatoires (non prédictibles) MAIS le drift entre eux l'est.
On ne prédit PAS le spike — on réagit APRÈS qu'il s'est produit.
"""
from __future__ import annotations

from deriv.constants import SPREAD_CRASH, SPREAD_BOOM
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class Direction(Enum):
    LONG  = "long"
    SHORT = "short"
    NONE  = "none"


@dataclass
class TradeSignal:
    timestamp:        object
    symbol:           str
    direction:        Direction
    entry_price:      float
    stop_loss:        float
    take_profit:      float
    regime:           str           # toujours "spike_drift"
    phase:            str           # "post_spike_entry"
    rationale:        str
    hmm_confidence:   float = 0.0   # non utilisé
    score:            float = 0.0
    pullback_candles: int = 0       # bars depuis le spike
    adx:              float = 0.0
    rsi:              float = 50.0
    atr:              float = 0.0
    spread_cost_pct:  float = 0.0   # coût du spread

    @property
    def risk_reward(self) -> float:
        if self.direction == Direction.LONG:
            risk   = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk   = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return round(reward / max(risk, 1e-6), 2)


class BoomCrashDriftStrategy:
    """
    Exploite le drift structurel entre les spikes Boom/Crash.

    Ne prédit PAS le spike — réagit après qu'il s'est produit.
    Edge réel : après un spike Boom, le prix DOIT redescendre
    (drift baissier de conception) avant le prochain spike.
    """

    # Seuils de détection de spike
    SPIKE_THRESHOLD_PCT = 0.5     # 0.5% de mouvement en 1 bougie = spike
    ADAPTIVE_K_STD       = 3.0    # sinon : 3×std de la fenêtre (données réelles)
    MIN_THRESHOLD_PCT    = 0.08   # seuil plancher (évite le bruit pur)
    POST_SPIKE_BARS     = 2       # attendre N bougies après le spike
    TARGET_RR           = 1.8     # TP = risque × RR (risque borné)

    def __init__(self, symbol: str = "BOOM500",
                 spike_threshold_pct: float | None = None):
        self.symbol           = symbol
        self.is_boom          = "BOOM" in symbol.upper()
        self.last_spike_time  = None
        self.last_spike_price = None
        self.bars_since_spike = 0
        # Seuil explicite (fixe) OU adaptatif si None
        self.spike_threshold_pct = spike_threshold_pct

    def _threshold(self, df: pd.DataFrame) -> float:
        """Seuil de spike : explicite si fourni, sinon adaptatif (3×std)."""
        if self.spike_threshold_pct is not None:
            return self.spike_threshold_pct
        if len(df) < 20:
            return self.SPIKE_THRESHOLD_PCT
        closes = df["close"].values
        moves = np.diff(closes) / closes[:-1] * 100.0
        std = float(np.std(moves))
        return max(self.MIN_THRESHOLD_PCT, self.ADAPTIVE_K_STD * std)

    # ── Détection du spike ─────────────────────────────────────────
    def detect_spike(self, df: pd.DataFrame) -> bool:
        """True si la DERNIÈRE bougie est un spike (>= seuil en 1 bougie)."""
        if len(df) < 2:
            return False
        last = df["close"].iloc[-1]
        prev = df["close"].iloc[-2]
        move = (last - prev) / prev * 100.0
        thr = self._threshold(df)
        if self.is_boom:
            return bool(move >= thr)
        return bool(move <= -thr)

    @staticmethod
    def _atr(df: pd.DataFrame) -> float:
        """ATR(14) sur la fenêtre reçue ; 0.0 si indisponible."""
        try:
            if len(df) < 15:
                return 0.0
            h, l, c = df["high"], df["low"], df["close"]
            atr = ta.volatility.AverageTrueRange(
                h, l, c, window=14).average_true_range()
            v = atr.iloc[-1]
            return 0.0 if pd.isna(v) else float(v)
        except Exception:
            return 0.0

    def analyse(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """
        Retourne un TradeSignal si un drift post-spike est exploitable.

        Algorithme :
          1. Spike sur la dernière bougie → mémorise, on attend N bougies.
          2. Si on est exactement POST_SPIKE_BARS bougies après le spike,
             on entre dans le sens du drift :
               BOOM  → SHORT (le prix redescend après le spike)
               CRASH → LONG  (le prix remonte après le spike)
          3. SL au-delà du spike ; TP = risque × TARGET_RR.
        """
        if len(df) < 20:
            return None

        # Spike sur la dernière bougie → mémorise, on attend
        if self.detect_spike(df):
            self.last_spike_time  = df.index[-1]
            self.last_spike_price = float(df["close"].iloc[-1])
            self.bars_since_spike = 0
            return None

        self.bars_since_spike += 1

        # Pas de spike connu → rien à trader
        if self.last_spike_price is None:
            return None

        if self.bars_since_spike != self.POST_SPIKE_BARS:
            return None

        close   = float(df["close"].iloc[-1])
        atr_val = self._atr(df)
        spike   = self.last_spike_price

        if self.is_boom:
            direction = Direction.SHORT
            if atr_val > 0:
                sl   = max(spike, close + atr_val)      # au-delà du spike haut
                risk = max(sl - close, atr_val * 0.5)
            else:
                sl   = spike * 1.002
                risk = sl - close
            tp = close - risk * self.TARGET_RR
        else:
            direction = Direction.LONG
            if atr_val > 0:
                sl   = min(spike, close - atr_val)      # en dessous du spike bas
                risk = max(close - sl, atr_val * 0.5)
            else:
                sl   = spike * 0.998
                risk = close - sl
            tp = close + risk * self.TARGET_RR

        ts = df.index[-1] if hasattr(df.index[-1], "isoformat") else None
        bars_since = self.bars_since_spike   # sauve avant reset
        rationale = (
            f"Drift post-spike {self.symbol} | "
            f"spike@{spike:.5f} | {bars_since} bars | "
            f"ATR={atr_val:.5f} | RR={self.TARGET_RR}"
        )

        self._reset_spike()
        return TradeSignal(
            timestamp        = ts,
            symbol           = self.symbol,
            direction        = direction,
            entry_price      = close,
            stop_loss        = round(float(sl), 7),
            take_profit      = round(float(tp), 7),
            regime           = "spike_drift",
            phase            = "post_spike_entry",
            rationale        = rationale,
            pullback_candles = bars_since,
            atr              = round(atr_val, 7),
            spread_cost_pct  = SPREAD_CRASH,   # SPREAD_CRASH / SPREAD_BOOM (constants)
        )

    def _reset_spike(self) -> None:
        self.last_spike_time  = None
        self.last_spike_price = None
        self.bars_since_spike = 0

