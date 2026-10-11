"""Sécurité argent réel + gestionnaire de risque qui reprend après une pause."""
import asyncio
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

from deriv.bot_executor import DerivBotExecutor
from deriv.client import is_virtual_loginid
from deriv.risk_manager import RiskManager, TradeResult


# ── Compte réellement autorisé par le token ────────────────────────────
@pytest.mark.parametrize("loginid,virtuel", [
    ("VRTC1234567", True), ("vrtc99", True), ("VRW1001", True),
    ("CR1234567", False), ("MF100", False), ("", False), (None, False),
])
def test_detection_compte_virtuel(loginid, virtuel):
    assert is_virtual_loginid(loginid) is virtuel


def _bot_avec_compte(loginid, dry_run=False):
    bot = DerivBotExecutor(capital_usd=150.0, dry_run=dry_run)

    async def fetch():
        return loginid

    bot.conn = SimpleNamespace(fetch_loginid=fetch, is_demo=True)
    return bot


def test_token_reel_refuse_en_live_meme_si_env_dit_demo(monkeypatch):
    monkeypatch.setattr("deriv.bot_executor.Config.ALLOW_REAL", False)
    bot = _bot_avec_compte("CR1234567")
    with pytest.raises(RuntimeError, match="RÉEL"):
        asyncio.run(bot._verify_account())
    assert bot.conn.is_demo is False


def test_compte_indetermine_refuse_en_live(monkeypatch):
    monkeypatch.setattr("deriv.bot_executor.Config.ALLOW_REAL", False)
    with pytest.raises(RuntimeError, match="indéterminé"):
        asyncio.run(_bot_avec_compte(None)._verify_account())


def test_compte_demo_accepte(monkeypatch):
    monkeypatch.setattr("deriv.bot_executor.Config.ALLOW_REAL", False)
    bot = _bot_avec_compte("VRTC1234567")
    asyncio.run(bot._verify_account())
    assert bot.conn.is_demo is True


def test_compte_reel_tolere_en_dry_run_ou_si_autorise(monkeypatch):
    monkeypatch.setattr("deriv.bot_executor.Config.ALLOW_REAL", False)
    asyncio.run(_bot_avec_compte("CR1", dry_run=True)._verify_account())
    monkeypatch.setattr("deriv.bot_executor.Config.ALLOW_REAL", True)
    asyncio.run(_bot_avec_compte("CR1")._verify_account())


# ── Risque : pause puis reprise ────────────────────────────────────────
def _rm():
    rm = RiskManager(capital=100.0)
    rm.daily_stop_pct, rm.max_consecutive = 0.5, 3
    return rm


def test_reprend_apres_la_pause_de_3_pertes():
    rm = _rm()
    for _ in range(3):
        rm.record(TradeResult("BOOM500", -1.0))
    assert rm.can_trade()[0] is False                      # pause active
    rm.cooldown_until = datetime.now() - timedelta(seconds=1)
    ok, _ = rm.can_trade()
    assert ok and rm.consecutive_loss == 0                 # avant : bloqué à vie


def test_pertes_consecutives_sans_cooldown_declenchent_une_pause():
    rm = _rm()
    rm.consecutive_loss = 3                                # état restauré
    assert rm.can_trade()[0] is False and rm.cooldown_until is not None


def test_stop_journalier_remis_a_zero_le_lendemain():
    rm = _rm()
    rm.daily_loss = 60.0
    assert rm.can_trade()[0] is False
    rm._day = date.today() - timedelta(days=1)
    assert rm.can_trade()[0] is True and rm.daily_loss == 0.0


def test_alerte_telegram_une_seule_fois_par_blocage(monkeypatch):
    bot = DerivBotExecutor(capital_usd=150.0)
    messages = []

    async def tg(msg):
        messages.append(msg)

    monkeypatch.setattr(bot, "_telegram", tg)
    bot.risk.consecutive_loss = bot.risk.max_consecutive
    assert asyncio.run(bot._risk_ok()) is False
    assert asyncio.run(bot._risk_ok()) is False
    assert len(messages) == 1 and "pause" in messages[0].lower()
