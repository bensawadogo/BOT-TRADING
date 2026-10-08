"""Le bot live trade la stratégie drift validée sur BOOM/CRASH."""
import asyncio

import numpy as np
import pandas as pd
import pytest

from deriv.bot_executor import DerivBotExecutor


@pytest.mark.parametrize("mode,symbol,attendu", [
    ("auto", "BOOM500", "drift"), ("auto", "CRASH1000", "drift"),
    ("auto", "frxEURUSD", "ensemble"), ("ensemble", "BOOM500", "ensemble"),
    ("drift", "BOOM500", "drift"), ("", "R_75", "ensemble"),
])
def test_choix_de_la_strategie(mode, symbol, attendu):
    assert DerivBotExecutor.resolve_strategy_mode(mode, symbol) == attendu


def _bot_drift(monkeypatch, dry_run=False):
    monkeypatch.setattr(DerivBotExecutor, "SYMBOL", "BOOM500")
    monkeypatch.setattr(DerivBotExecutor, "GRANULARITY", 60)
    monkeypatch.setattr("deriv.bot_executor.Config.STRATEGY", "auto")
    return DerivBotExecutor(capital_usd=150.0, dry_run=dry_run)


def _historique_avec_spike(n=1502):
    """n bougies fermées (dernière = spike haussier) + 1 bougie en cours."""
    rng = np.random.default_rng(3)
    closes = list(1000 * np.cumprod(1 + rng.normal(-0.002, 0.01, n - 1) / 100))
    closes.append(closes[-1] * 1.01)                     # spike +1 %
    closes.append(closes[-1])                            # bougie EN COURS
    fin = pd.Timestamp.now(tz="UTC").floor("min")
    idx = pd.date_range(end=fin, periods=n + 1, freq="min")
    return pd.DataFrame({"close": closes}, index=idx)


class FauxCollecteur:
    def __init__(self, df):
        self.df, self.demandes = df, []

    async def get_candle_history(self, count):
        self.demandes.append(count)
        return self.df


def test_mode_drift_expiration_validee(monkeypatch):
    bot = _bot_drift(monkeypatch)
    assert bot.strategy_mode == "drift"
    assert bot.DURATION == 600                            # 10 bougies M1


def test_cycle_drift_achete_un_put_apres_spike_boom(monkeypatch):
    bot = _bot_drift(monkeypatch)
    bot.collector = FauxCollecteur(_historique_avec_spike())
    achats = []

    async def faux_execute(contract_type, journaliser=None):
        achats.append(contract_type)

    monkeypatch.setattr(bot, "_execute_contract", faux_execute)
    asyncio.run(bot._run_drift_cycle())
    assert achats == ["PUT"]
    assert bot.collector.demandes == [bot.drift.min_bars + 1]
    asyncio.run(bot._run_drift_cycle())                   # même bougie : pas de 2e achat
    assert achats == ["PUT"]


def test_cycle_drift_dry_run_n_achete_rien(monkeypatch, caplog):
    bot = _bot_drift(monkeypatch, dry_run=True)
    bot.collector = FauxCollecteur(_historique_avec_spike())

    async def interdit(*a, **k):
        pytest.fail("achat en dry-run")

    monkeypatch.setattr(bot, "_execute_contract", interdit)
    with caplog.at_level("INFO", logger="DERIV_BOT"):
        asyncio.run(bot._run_drift_cycle())
    assert "DRY-RUN : PUT" in caplog.text


def test_journal_drift_complet(monkeypatch, tmp_path):
    from deriv.trading_journal import TradingJournal
    from deriv.strategies.spike_drift_binary import SpikeSignal

    bot = _bot_drift(monkeypatch)
    bot.journal = TradingJournal(db_path=str(tmp_path / "j.db"))
    sig = SpikeSignal("PUT", None, 1000.0, 1.0, 0.3, "test")
    trade_id = asyncio.run(bot._journaliser_drift(sig))
    assert trade_id in bot.open_trades
    res = bot.journal.log_exit_contract(trade_id, pnl_dollar=1.6)
    assert res["pnl_net_pct"] == pytest.approx(80.0)      # pas de spread déduit
