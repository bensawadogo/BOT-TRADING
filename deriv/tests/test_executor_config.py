"""Le bot live doit tourner dans la configuration où il a été validé."""
from deriv.bot_executor import DerivBotExecutor
from deriv.constants import Config


def test_collecteur_utilise_la_granularite_pas_la_duree(monkeypatch):
    monkeypatch.setattr(DerivBotExecutor, "GRANULARITY", 60)
    monkeypatch.setattr(DerivBotExecutor, "DURATION", 240)
    bot = DerivBotExecutor(capital_usd=150.0)
    collector = bot._make_collector()
    assert collector.tick_size == 60          # bougies M1, comme au WFO
    assert collector.tick_size != bot.DURATION


def test_granularite_par_defaut_vient_de_la_config():
    assert DerivBotExecutor.GRANULARITY == Config.GRANULARITY
