"""
Stratégie VALIDÉE : drift post-spike Boom/Crash en contrat Rise/Fall.

Source unique de vérité pour la validation (deriv/validate_boom_crash_big.py)
ET pour le bot live (deriv/bot_executor.py) : les deux appellent les mêmes
fonctions, le live trade donc exactement ce qui a été mesuré.

Règle (WFO 13/09/2026, BOOM500 q99 : 62.5 % sur 104 trades OOS, p = 0.093) :
  1. Seuil de spike = quantile q des |mouvements| des `calib_window` bougies
     PRÉCÉDENTES (jamais la bougie testée → aucune fuite).
  2. Si la dernière bougie fermée est un spike (BOOM : hausse >= seuil,
     CRASH : baisse <= -seuil), on parie sur le drift inverse :
       BOOM → PUT (Fall), CRASH → CALL (Rise).
  3. Expiration = `horizon` bougies (10 × M1 = 600 s).

Ce n'est PAS la classe BoomCrashDriftStrategy (seuil 3×std, entrée 2 bougies
après le spike, SL/TP) : cette variante-là n'a jamais été validée.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

QUANTILE = 0.99          # q99 : meilleur compromis trades / WR mesuré
CALIB_WINDOW = 1500      # = fold_size du walk-forward
HORIZON = 10             # bougies jusqu'à l'expiration
MIN_THRESHOLD_PCT = 0.08
MAX_OPEN_CONTRACTS = 3   # contrats simultanés max (spikes en grappe)


def spike_threshold_quantile(closes: np.ndarray, is_boom: bool = True,
                             q: float = QUANTILE,
                             min_thr: float = MIN_THRESHOLD_PCT) -> float:
    """Seuil de spike = quantile q de |moves| en % (évite le bruit du 3×std).

    `is_boom` est conservé pour compatibilité : le seuil est symétrique.
    """
    closes = np.asarray(closes, dtype=float)
    moves = np.abs(np.diff(closes) / closes[:-1] * 100.0)
    if moves.size < 50:
        return min_thr
    return max(min_thr, float(np.quantile(moves, q)))


def detect_spikes_idx(closes: np.ndarray, thr: float,
                      is_boom: bool) -> np.ndarray:
    """Indices i où la bougie i-1 → i est un spike (|move| >= thr)."""
    closes = np.asarray(closes, dtype=float)
    moves = np.diff(closes) / closes[:-1] * 100.0
    if is_boom:
        return np.nonzero(moves >= thr)[0] + 1
    return np.nonzero(moves <= -thr)[0] + 1


def closed_candles(df: pd.DataFrame, granularity: int,
                   now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Retire la bougie encore en cours (index = heure d'OUVERTURE, Deriv)."""
    if df is None or df.empty:
        return df
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    still_open = idx + pd.Timedelta(seconds=int(granularity)) > now
    return df[~still_open]


@dataclass
class SpikeSignal:
    contract_type: str       # "CALL" (Rise) ou "PUT" (Fall)
    spike_time: object
    entry_price: float
    move_pct: float
    threshold_pct: float
    rationale: str


class SpikeDriftBinaryStrategy:
    """Évalue la dernière bougie FERMÉE ; une seule décision par bougie."""

    def __init__(self, symbol: str = "BOOM500", quantile: float = QUANTILE,
                 calib_window: int = CALIB_WINDOW, horizon: int = HORIZON):
        self.symbol = symbol
        self.is_boom = "BOOM" in symbol.upper()
        self.quantile = quantile
        self.calib_window = calib_window
        self.horizon = horizon
        self._last_evaluated = None

    @property
    def min_bars(self) -> int:
        """Bougies nécessaires : fenêtre de calibration + la bougie testée."""
        return self.calib_window + 1

    def contract_duration(self, granularity: int) -> int:
        """Durée du contrat en secondes (horizon bougies)."""
        return int(self.horizon * granularity)

    def evaluate(self, df: pd.DataFrame) -> SpikeSignal | None:
        if df is None or len(df) < self.min_bars or "close" not in df:
            return None
        ts = df.index[-1]
        if ts == self._last_evaluated:      # déjà décidée
            return None
        self._last_evaluated = ts

        closes = df["close"].to_numpy(dtype=float)
        # Calibration sur les bougies AVANT la bougie testée uniquement.
        calib = closes[-(self.calib_window + 1):-1]
        thr = spike_threshold_quantile(calib, self.is_boom, q=self.quantile)
        if detect_spikes_idx(closes[-2:], thr, self.is_boom).size == 0:
            return None

        move = (closes[-1] - closes[-2]) / closes[-2] * 100.0
        contract = "PUT" if self.is_boom else "CALL"
        return SpikeSignal(
            contract_type=contract,
            spike_time=ts,
            entry_price=float(closes[-1]),
            move_pct=float(move),
            threshold_pct=float(thr),
            rationale=(f"Spike {self.symbol} {move:+.3f}% (seuil q{self.quantile:g} "
                       f"{thr:.3f}%) → {contract} {self.horizon} bougies"),
        )
