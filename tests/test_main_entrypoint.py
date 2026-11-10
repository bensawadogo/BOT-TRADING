"""main.py doit lancer le bot de tendance, en simulation par défaut."""
import pytest

import main as entry


@pytest.fixture
def appels(monkeypatch):
    calls = {}

    class FauxBroker:
        def __init__(self):
            calls["broker"] = True

    monkeypatch.setattr("trend_bot.brokers.mt5.MT5Broker", FauxBroker)
    monkeypatch.setattr("trend_bot.runner.check_account",
                        lambda broker, live: calls.setdefault("live", live))
    monkeypatch.setattr("trend_bot.runner.run_cycle",
                        lambda broker, dry_run, marches=None: calls.setdefault("dry_run", dry_run))
    monkeypatch.setattr("trend_bot.runner.loop",
                        lambda broker, dry_run: calls.setdefault("loop", dry_run))
    monkeypatch.setattr(entry.subprocess, "run", lambda *a, **k: calls.setdefault("dashboard", a))
    return calls


def test_sans_argument_un_cycle_en_simulation(appels):
    entry.main([])
    assert appels == {"broker": True, "live": False, "dry_run": True}


def test_live_envoie_les_ordres(appels):
    entry.main(["trend", "--live"])
    assert appels["live"] is True and appels["dry_run"] is False


def test_loop_reste_en_simulation_sans_live(appels):
    entry.main(["trend", "--loop"])
    assert appels["loop"] is True and "dry_run" not in appels


def test_dashboard(appels):
    entry.main(["dashboard"])
    assert "dashboard" in appels and "broker" not in appels
