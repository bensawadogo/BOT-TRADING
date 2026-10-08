"""
════════════════════════════════════════════════════════════════════
TESTS STRATÉGIE BOOM/CRASH DRIFT
deriv/tests/test_boom_crash.py
════════════════════════════════════════════════════════════════════

Valide :
  1. Détection de spike (Boom = +0.5%, Crash = -0.5%)
  2. Entrée POST_SPIKE_BARS bougies après le spike (SHORT sur Boom)
  3. Drift baissier après spike Boom → SHORT (SL au-dessus du spike)
  4. Drift haussier après spike Crash → LONG (SL en dessous du spike)
  5. Le journal soustrait le spread (pnl_net_pct = pnl_pct − spread)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import pytest

from deriv.strategies.boom_crash_drift import (
    BoomCrashDriftStrategy,
    Direction,
    TradeSignal,
)
from deriv.trading_journal import JournalEntry, TradingJournal


# ── Fixtures ───────────────────────────────────────────────────────

def make_boom_df(n: int = 60, spike_at: int = 50, drop_pct: float = 0.010) -> pd.DataFrame:
    """Série Boom : spike haussier puis drift baissier post-spike."""
    np.random.seed(42)
    t = np.arange(n)
    base = 1000 + t * 0.005
    close = base.copy().astype(float)
    close[spike_at] = close[spike_at - 1] * (1 + 0.006)  # +0.6%
    for i in range(spike_at + 1, n):
        close[i] = close[i - 1] * (1 - drop_pct) + np.random.randn() * 0.1
    open_ = close.copy()
    high = close + 2.0
    low = close - 2.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.ones(n) * 100,
    })
    df.index = pd.date_range("2026-09-01", periods=n, freq="1min")
    return df


def make_crash_df(n: int = 60, spike_at: int = 50, rise_pct: float = 0.010) -> pd.DataFrame:
    """Série Crash : spike baissier puis drift haussier post-spike."""
    np.random.seed(7)
    t = np.arange(n)
    base = 5000 - t * 0.005
    close = base.copy().astype(float)
    close[spike_at] = close[spike_at - 1] * (1 - 0.006)  # -0.6%
    for i in range(spike_at + 1, n):
        close[i] = close[i - 1] * (1 + rise_pct) + np.random.randn() * 0.1
    open_ = close.copy()
    high = close + 2.0
    low = close - 2.0
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": np.ones(n) * 100,
    })
    df.index = pd.date_range("2026-09-01", periods=n, freq="1min")
    return df


# ── Tests détection de spike ───────────────────────────────────────

class TestDetectSpike:
    def test_boom_detecte_spike_haussier(self):
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        assert s.detect_spike(df.iloc[:51])  # +0.6% > +0.5%

    def test_boom_pas_de_spike_sur_drift(self):
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        assert not s.detect_spike(df.iloc[:53])

    def test_crash_detecte_spike_baissier(self):
        s = BoomCrashDriftStrategy("CRASH500")
        df = make_crash_df()
        assert s.detect_spike(df.iloc[:51])

    def test_crash_pas_de_spike_sur_drift(self):
        s = BoomCrashDriftStrategy("CRASH500")
        df = make_crash_df()
        assert not s.detect_spike(df.iloc[:53])

    def test_moins_de_2_bougies_pas_de_spike(self):
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        assert not s.detect_spike(df.iloc[:1])


# ── Tests machine à états / analyse ────────────────────────────────

class TestAnalyse:
    def test_entree_short_sur_boom(self):
        """Après spike Boom → SHORT avec SL au-dessus du spike."""
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        signal = None
        for i in range(50, 58):
            signal = s.analyse(df.iloc[:i + 1])
            if signal is not None:
                break
        assert signal is not None, "Aucun signal émis après le spike"
        assert signal.direction == Direction.SHORT
        assert signal.stop_loss > signal.entry_price

    def test_entree_long_sur_crash(self):
        """Après spike Crash → LONG avec SL en dessous du spike."""
        s = BoomCrashDriftStrategy("CRASH500")
        df = make_crash_df()
        signal = None
        for i in range(50, 58):
            signal = s.analyse(df.iloc[:i + 1])
            if signal is not None:
                break
        assert signal is not None, "Aucun signal émis après le spike"
        assert signal.direction == Direction.LONG
        assert signal.stop_loss < signal.entry_price

    def test_signal_a_pullback_candles(self):
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        for i in range(50, 58):
            sig = s.analyse(df.iloc[:i + 1])
            if sig is not None:
                assert sig.pullback_candles == s.POST_SPIKE_BARS
                return
        pytest.fail("Aucun signal")

    def test_reset_apres_signal(self):
        """Après un signal, la stratégie doit être prête pour le prochain spike."""
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        for i in range(50, 58):
            sig = s.analyse(df.iloc[:i + 1])
            if sig is not None:
                break
        assert s.last_spike_price is None
        assert s.bars_since_spike == 0

    def test_pas_de_signal_sans_spike(self):
        # Série plate : aucun mouvement ≥ 0.5% → aucun spike → aucun signal
        np.random.seed(3)
        n = 60
        close = 1000 + np.random.randn(n) * 0.1
        open_ = close.copy()
        df = pd.DataFrame({
            "open": open_, "high": close + 0.2, "low": close - 0.2,
            "close": close, "volume": np.ones(n) * 100,
        })
        df.index = pd.date_range("2026-09-01", periods=n, freq="1min")
        s = BoomCrashDriftStrategy("BOOM500")
        for i in range(0, 58):
            sig = s.analyse(df.iloc[:i + 1])
            assert sig is None, f"Signal émis sans spike : {sig}"
        assert s.last_spike_price is None

    def test_signal_spread_cost_0_2(self):
        s = BoomCrashDriftStrategy("BOOM500")
        df = make_boom_df()
        for i in range(50, 58):
            sig = s.analyse(df.iloc[:i + 1])
            if sig is not None:
                assert sig.spread_cost_pct == pytest.approx(0.2)
                return
        pytest.fail("Aucun signal")


# ── Tests journal : spread / pnl net ───────────────────────────────

def _make_entry(trade_id: str, symbol: str = "BOOM500",
                direction: str = "short", spread: float = 0.2) -> JournalEntry:
    return JournalEntry(
        trade_id=trade_id, timestamp_entry="2026-09-12T04:00:00+00:00",
        symbol=symbol, direction=direction,
        entry_price=1000.0, stop_loss=1002.0, take_profit=997.0,
        risk_reward=1.8, stake=1.0,
        regime="spike_drift", hmm_confidence=0.0, adx=0.0, rsi=50.0,
        atr=1.0, ema_fast=1000.0, ema_slow=1000.0, volume_ratio=1.0,
        session="london",
        vote_hmm=1, vote_xgboost=1, vote_lstm=0, vote_kalman=1,
        vote_rsi=1, vote_trend=0, vote_momentum=1,
        ensemble_votes=5, ensemble_signal="SELL",
        strategy_phase="post_spike_entry", pullback_candles=2,
        rationale="Drift post-spike BOOM500",
        spread_cost_pct=spread,
    )


class TestJournalSpread:
    def test_pnl_net_soustrait_le_spread(self, tmp_path):
        j = TradingJournal(db_path=str(tmp_path / "t.db"))
        j.log_entry(_make_entry("BC001"))
        res = j.log_exit("BC001", exit_price=998.0, exit_reason="tp")
        # brut = (1000−998)/1000×100 = +0.2% ; net = +0.2 − 0.2 = 0.0
        assert res["pnl_pct"] == pytest.approx(0.2, abs=0.001)
        assert res["pnl_net_pct"] == pytest.approx(0.0, abs=0.001)

    def test_pnl_net_negatif_si_spread_gros(self, tmp_path):
        j = TradingJournal(db_path=str(tmp_path / "t2.db"))
        j.log_entry(_make_entry("BC002", spread=0.5))
        res = j.log_exit("BC002", exit_price=999.8, exit_reason="tp")
        # brut = +0.02% ; net = +0.02 − 0.5 → négatif
        assert res["pnl_net_pct"] < 0

    def test_pnl_net_zero_sans_spread(self, tmp_path):
        j = TradingJournal(db_path=str(tmp_path / "t3.db"))
        j.log_entry(_make_entry("BC003", spread=0.0))
        res = j.log_exit("BC003", exit_price=998.0, exit_reason="tp")
        assert res["pnl_net_pct"] == pytest.approx(0.2, abs=0.001)


# ── Tests du signal TradeSignal ────────────────────────────────────

class TestTradeSignal:
    def test_risk_reward_short(self):
        s = TradeSignal(
            timestamp=None, symbol="BOOM500", direction=Direction.SHORT,
            entry_price=1000.0, stop_loss=1002.0, take_profit=996.4,
            regime="spike_drift", phase="post_spike_entry", rationale="",
        )
        # RR = (1000−996.4)/(1002−1000) = 3.6/2 = 1.8
        assert s.risk_reward == pytest.approx(1.8)

    def test_risk_reward_long(self):
        s = TradeSignal(
            timestamp=None, symbol="CRASH500", direction=Direction.LONG,
            entry_price=5000.0, stop_loss=4998.0, take_profit=5003.6,
            regime="spike_drift", phase="post_spike_entry", rationale="",
        )
        assert s.risk_reward == pytest.approx(1.8)