"""Journal cohérent, apprentissage online propre, rapport non spammé."""
import asyncio
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from deriv.bot_executor import DerivBotExecutor
from deriv.strategies.spike_drift_binary import PHASE, SpikeSignal
from deriv.trading_journal import TradingJournal


@pytest.fixture
def bot(tmp_path):
    b = DerivBotExecutor(capital_usd=150.0)
    b.journal = TradingJournal(db_path=str(tmp_path / "j.db"))
    return b


def test_plus_de_strategie_non_validee_dans_le_bot(bot):
    assert not hasattr(bot, "boom_crash")


def test_journal_ensemble_et_drift_meme_format(bot):
    idx = pd.date_range("2026-01-05 12:00", periods=120, freq="min", tz="UTC")
    rng = np.random.default_rng(0)
    c = 100 + np.cumsum(rng.normal(0, 0.1, 120))
    df = pd.DataFrame({"open": c, "high": c + .1, "low": c - .1, "close": c,
                       "volume": 1.0}, index=idx)
    detail = {k: {"vote": 1.0} for k in ("HMM", "XGBoost", "LSTM", "Kalman",
                                         "RSI", "TrendStrength", "Momentum")}
    detail["HMM"].update(regime="Bull", confiance=0.9)
    result = {"signal": "BUY", "buy_votes": 5, "sell_votes": 0, "detail": detail}
    t_ens = asyncio.run(bot._journaliser_setup(df, result))
    t_drift = asyncio.run(bot._journaliser_drift(
        SpikeSignal("PUT", None, 1000.0, 1.0, 0.3, "spike")))
    assert t_ens and t_drift
    for tid in (t_ens, t_drift):
        e = bot.open_trades[tid]["entry"]
        assert e.spread_cost_pct == 0.0          # PnL Rise/Fall déjà net
        assert tid.endswith(bot.SYMBOL) and len(tid.split("_")[1]) == 6  # HHMMSS
    assert bot.open_trades[t_drift]["entry"].strategy_phase == PHASE


def _online_espion(bot):
    appels = []
    bot.online = SimpleNamespace(update=lambda feats, label: appels.append(label) or {})
    bot.learner = None
    return appels


def test_online_learner_ignore_les_trades_drift(bot):
    appels = _online_espion(bot)
    tid = asyncio.run(bot._journaliser_drift(SpikeSignal("PUT", None, 1.0, 1.0, .3, "s")))
    asyncio.run(bot._close_trade("PUT", tid, 1.6, 42))
    assert appels == []


def test_online_learner_apprend_des_trades_ensemble(bot):
    appels = _online_espion(bot)
    tid = bot._log_entry(direction="long", entry_price=1.0, strategy_phase="ENSEMBLE",
                         ensemble_signal="BUY", rationale="ens")
    asyncio.run(bot._close_trade("CALL", tid, 1.6, 42))
    assert appels == [1]


def test_rapport_learner_au_plus_une_fois_par_semaine(bot):
    envois = []

    async def tg(msg):
        envois.append(msg)

    bot._telegram = tg
    bot.learner = SimpleNamespace(analyse_et_recommande=lambda: {
        "profitable": True, "n_trades_analyses": 50, "message_telegram": "rapport"})
    for _ in range(5):
        asyncio.run(bot._rapport_periodique())
    assert envois == ["rapport"]
    bot._last_report_at = datetime.now() - timedelta(days=8)
    asyncio.run(bot._rapport_periodique())
    assert envois == ["rapport", "rapport"]
