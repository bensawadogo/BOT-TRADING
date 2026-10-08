"""Le bot live trade la stratégie drift validée sur BOOM/CRASH."""
import asyncio

from types import SimpleNamespace

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


FIN = pd.Timestamp("2026-03-02 10:00", tz="UTC")   # ouverture de la bougie en cours


def _bot_drift(monkeypatch, dry_run=False, serveur=FIN + pd.Timedelta(seconds=3)):
    monkeypatch.setattr(DerivBotExecutor, "SYMBOL", "BOOM500")
    monkeypatch.setattr(DerivBotExecutor, "GRANULARITY", 60)
    monkeypatch.setattr(DerivBotExecutor, "FETCH_RETRY_SEC", 0)
    monkeypatch.setattr("deriv.bot_executor.Config.STRATEGY", "auto")
    bot = DerivBotExecutor(capital_usd=150.0, dry_run=dry_run)

    async def server_time():
        return serveur.timestamp()

    bot.conn = SimpleNamespace(server_time=server_time)
    return bot


def _historique_avec_spike(n=1502):
    """n bougies fermées (dernière = spike haussier) + 1 bougie en cours."""
    rng = np.random.default_rng(3)
    closes = list(1000 * np.cumprod(1 + rng.normal(-0.002, 0.01, n - 1) / 100))
    closes.append(closes[-1] * 1.01)                     # spike +1 %
    closes.append(closes[-1])                            # bougie EN COURS
    idx = pd.date_range(end=FIN, periods=n + 1, freq="min")
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

    async def deux_cycles():
        t = await bot._run_drift_cycle()
        await t
        assert await bot._run_drift_cycle() is None      # même bougie : pas de 2e achat

    asyncio.run(deux_cycles())
    assert achats == ["PUT"]
    assert bot.collector.demandes[0] == bot.drift.min_bars + 1


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


def test_horloge_locale_en_avance_bougie_pas_encore_fermee(monkeypatch):
    """Le serveur n'a pas fini la bougie du spike : elle ne doit pas être jugée."""
    df = _historique_avec_spike().iloc[:-1]           # dernière = bougie du spike
    serveur = df.index[-1] + pd.Timedelta(seconds=58)  # serveur : 2 s avant la clôture
    bot = _bot_drift(monkeypatch, serveur=serveur)
    bot.collector = FauxCollecteur(df)
    achats = []

    async def faux_execute(contract_type, journaliser=None):
        achats.append(contract_type)

    monkeypatch.setattr(bot, "_execute_contract", faux_execute)
    assert asyncio.run(bot._run_drift_cycle()) is None
    assert achats == [] and bot.drift._last_evaluated != df.index[-1]


def test_l_analyse_continue_pendant_un_contrat(monkeypatch):
    """Un 2e spike pendant un contrat ouvert est tradé (comme au WFO)."""
    bot = _bot_drift(monkeypatch)
    df = _historique_avec_spike()
    bot.collector = FauxCollecteur(df)
    liberer = None

    async def contrat_long(contract_type, journaliser=None):
        await liberer.wait()

    monkeypatch.setattr(bot, "_execute_contract", contrat_long)

    async def scenario():
        nonlocal liberer
        liberer = asyncio.Event()
        t1 = await bot._run_drift_cycle()
        assert t1 is not None and len(bot._open_contracts) == 1
        # Bougie suivante : nouveau spike pendant que le 1er contrat tourne
        suivante = df.index[-1] + pd.Timedelta(minutes=1)
        df2 = pd.concat([df.iloc[:-1], pd.DataFrame(
            {"close": [df["close"].iloc[-2] * 1.01, df["close"].iloc[-2] * 1.01]},
            index=[df.index[-1], suivante])])
        bot.collector.df = df2

        async def plus_tard():
            return (suivante + pd.Timedelta(seconds=3)).timestamp()

        bot.conn.server_time = plus_tard
        t2 = await bot._run_drift_cycle()
        assert t2 is not None and len(bot._open_contracts) == 2
        liberer.set()
        await asyncio.gather(t1, t2)
        await asyncio.sleep(0)
        assert bot._open_contracts == set()

    asyncio.run(scenario())


def test_limite_de_contrats_simultanes(monkeypatch):
    from deriv.bot_executor import MAX_OPEN_CONTRACTS

    bot = _bot_drift(monkeypatch)
    bot.collector = FauxCollecteur(_historique_avec_spike())
    bot._open_contracts = {object() for _ in range(MAX_OPEN_CONTRACTS)}

    async def interdit(*a, **k):
        pytest.fail("contrat au-delà de la limite")

    monkeypatch.setattr(bot, "_execute_contract", interdit)
    assert asyncio.run(bot._run_drift_cycle()) is None
