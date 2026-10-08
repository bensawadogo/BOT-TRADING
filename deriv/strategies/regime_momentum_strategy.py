"""
════════════════════════════════════════════════════════════════════
STRATÉGIE EDGE — Régime + Momentum (machine à états 4 phases)
deriv/strategies/regime_momentum_strategy.py
════════════════════════════════════════════════════════════════════

Stratégie basée sur les recherches académiques (HMM régime + pullback) :
1. HMM détecte le régime (Bull/Bear/Range)
2. Entrer UNIQUEMENT en régime Bull avec momentum confirmé
3. Machine à états 4 phases (SCANNING → ARMED → WINDOW → ENTRY)
4. Pas de trade en Bear ou Range → moins de trades, meilleur WR

Résultats académiques de référence (XAUUSD H4) :
  - Régime Bull  → Sharpe 3.82, return 4.49% annualisé
  - Régime Bear  → Sharpe -0.02 (ne pas trader)
  - Régime Range → Sharpe 2.03 (neutre)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd
import ta

logger = logging.getLogger(__name__)


class Phase(Enum):
    SCANNING = "scanning"    # Surveillance, pas d'opportunité
    ARMED    = "armed"       # Signal détecté, attend pullback
    WINDOW   = "window"      # Pullback confirmé, attend breakout
    ENTRY    = "entry"       # Trade exécuté


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
    regime:           str           # Bull / Bear / Range
    hmm_confidence:   float         # 0-1
    score:            float         # 0-100
    phase:            str           # quelle phase a déclenché
    rationale:        str
    pullback_candles: int = 0       # nb de bougies de pullback avant breakout
    adx:              float = 0.0
    rsi:              float = 50.0
    atr:              float = 0.0

    @property
    def risk_reward(self) -> float:
        if self.direction == Direction.LONG:
            risk   = self.entry_price - self.stop_loss
            reward = self.take_profit - self.entry_price
        else:
            risk   = self.stop_loss - self.entry_price
            reward = self.entry_price - self.take_profit
        return round(reward / max(risk, 1e-6), 2)


class RegimeMomentumStrategy:
    """
    Machine à états 4 phases inspirée de la recherche XAUUSD.
    Ne trade QUE si :
      1. HMM dit Bull avec confiance > 70%
      2. Momentum EMA confirme (alignement haussier)
      3. Pullback de 1-3 bougies (entrée optimale)
      4. Breakout du high du pullback
    """

    EMA_FAST   = 14
    EMA_MEDIUM = 18
    EMA_SLOW   = 24
    PULLBACK_MIN = 1
    PULLBACK_MAX = 3
    ATR_STOP_MULT   = 1.5   # Stop = 1.5 × ATR
    ATR_TARGET_MULT = 2.5   # Target = 2.5 × ATR (RR = 1.67)
    MIN_HMM_CONFIDENCE = 0.70
    MIN_ADX            = 20  # Trend strength minimum

    def __init__(self):
        self.phase          = Phase.SCANNING
        self.direction      = Direction.NONE
        self.armed_price: Optional[float] = None
        self.pullback_count = 0
        self.window_high: Optional[float] = None
        self.window_low:  Optional[float] = None

    # ── Analyse principale ──────────────────────────────────────────
    def analyse(self, df: pd.DataFrame, hmm_result: dict) -> Optional[TradeSignal]:
        """
        Analyse les données et retourne un TradeSignal si opportunité.
        df : OHLCV, au moins 50 bougies
        hmm_result : {"regime": ..., "confiance": ...} (sortie de HMMPredictor)
        """
        if df is None or len(df) < 50:
            return None

        regime     = hmm_result.get("regime", "Range")
        confidence = float(hmm_result.get("confiance", 0.0))

        # RÈGLE 1 : NE PAS TRADER en Bear
        # (recherche : Sharpe -0.02 en Bear → perdant systématique)
        if regime == "Bear":
            self._reset()
            return None

        # RÈGLE 2 : Confiance HMM insuffisante → pas de trade
        if confidence < self.MIN_HMM_CONFIDENCE:
            self._reset()
            return None

        c = df["close"]
        h = df["high"]
        l = df["low"]

        ema_f = ta.trend.EMAIndicator(c, self.EMA_FAST).ema_indicator()
        ema_m = ta.trend.EMAIndicator(c, self.EMA_MEDIUM).ema_indicator()
        ema_s = ta.trend.EMAIndicator(c, self.EMA_SLOW).ema_indicator()
        atr   = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
        adx   = ta.trend.ADXIndicator(h, l, c, 14).adx()
        rsi   = ta.momentum.RSIIndicator(c, 14).rsi()

        curr_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else 0.0
        curr_adx = float(adx.iloc[-1]) if pd.notna(adx.iloc[-1]) else 0.0
        curr_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0

        # RÈGLE 3 : ADX minimum (marché trending)
        if curr_adx < self.MIN_ADX:
            return None

        curr_close = float(c.iloc[-1])
        prev_close = float(c.iloc[-2])
        curr_ema_f = float(ema_f.iloc[-1])
        curr_ema_m = float(ema_m.iloc[-1])
        curr_ema_s = float(ema_s.iloc[-1])

        # ── MACHINE À ÉTATS ─────────────────────────────────────────

        if self.phase == Phase.SCANNING:
            # PHASE 1 : SCANNING → ARMED
            ema_bull = curr_ema_f > curr_ema_m > curr_ema_s
            if regime == "Bull" and ema_bull:
                self.phase          = Phase.ARMED
                self.direction      = Direction.LONG
                self.armed_price    = curr_close
                self.pullback_count = 0
            return None

        if self.phase == Phase.ARMED:
            # PHASE 2 : ARMED → WINDOW (pullback de 1-3 bougies)
            if curr_close < prev_close:
                self.pullback_count += 1
                if self.PULLBACK_MIN <= self.pullback_count <= self.PULLBACK_MAX:
                    window_slice = df.iloc[-(self.pullback_count + 1):]
                    self.window_high = float(window_slice["high"].max())
                    self.window_low  = float(window_slice["low"].min())
                    self.phase       = Phase.WINDOW
                elif self.pullback_count > self.PULLBACK_MAX:
                    self._reset()   # pullback trop long → setup raté
            # bougie haussière : toujours ARMED, on attend le pullback
            return None

        if self.phase == Phase.WINDOW:
            # PHASE 3 : WINDOW → ENTRY (breakout)
            if curr_close < prev_close:
                # Pullback qui continue : on étend la window (borné)
                self.pullback_count += 1
                if self.pullback_count > self.PULLBACK_MAX + 1:
                    self._reset()
                    return None
                window_slice = df.iloc[-(self.pullback_count + 1):]
                self.window_high = float(window_slice["high"].max())
                self.window_low  = float(window_slice["low"].min())
                return None

            if self.window_high is not None and curr_close > self.window_high:
                # SIGNAL D'ENTRÉE
                entry = curr_close
                sl    = entry - curr_atr * self.ATR_STOP_MULT
                tp    = entry + curr_atr * self.ATR_TARGET_MULT
                pullback = self.pullback_count
                window_high = self.window_high

                rationale = (
                    f"HMM {regime} ({confidence:.0%}) | "
                    f"EMA alignées | ADX {curr_adx:.1f} | "
                    f"Pullback {pullback}B | "
                    f"Breakout au-dessus {window_high:.5f}"
                )
                signal = TradeSignal(
                    timestamp        = df.index[-1],
                    symbol           = "XAUUSD",
                    direction        = Direction.LONG,
                    entry_price      = entry,
                    stop_loss        = sl,
                    take_profit      = tp,
                    regime           = regime,
                    hmm_confidence   = confidence,
                    score            = confidence * 100,
                    phase            = "ENTRY_BREAKOUT",
                    rationale        = rationale,
                    pullback_candles = pullback,
                    adx              = curr_adx,
                    rsi              = curr_rsi,
                    atr              = curr_atr,
                )
                self._reset()   # prêt pour le prochain trade
                return signal

        return None

    def _reset(self):
        self.phase          = Phase.SCANNING
        self.direction      = Direction.NONE
        self.armed_price    = None
        self.pullback_count = 0
        self.window_high    = None
        self.window_low     = None
