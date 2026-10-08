"""Tests unitaires et d'intégration pour le bot Deriv."""

import os
import sys
import unittest

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deriv.client import DerivConnection, DerivClient, BalanceWrapper
from deriv.strategies.rise_fall import RiseFallStrategy
from deriv.risk_manager import RiskManager
NETWORK_AVAILABLE = os.environ.get("RUN_NETWORK_TESTS", "0") == "1"


class TestDerivConnection(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.skipif(not NETWORK_AVAILABLE, reason="Nécessite connexion réseau Deriv")
    @pytest.mark.integration
    async def test_connect_and_ping(self):
        conn = DerivConnection(app_id="1089")
        await conn.connect()
        try:
            ping_res = await conn.ping()
            self.assertIn("ping", ping_res)
            self.assertEqual(ping_res["ping"], "pong")
        finally:
            await conn.close()

    @pytest.mark.skipif(not NETWORK_AVAILABLE, reason="Nécessite connexion réseau Deriv")
    @pytest.mark.integration
    async def test_get_ticks_live(self):
        conn = DerivConnection(app_id="1089")
        await conn.connect()
        try:
            history = await conn.get_ticks(symbol="R_75")
            self.assertTrue(hasattr(history, "candles"))
            self.assertGreater(len(history.candles), 0)
            first_candle = history.candles[0]
            self.assertTrue(hasattr(first_candle, "close"))
        finally:
            await conn.close()

    def test_balance_wrapper(self):
        b = BalanceWrapper(10000.0, currency="USD")
        self.assertEqual(float(b), 10000.0)
        self.assertEqual(float(b.balance), 10000.0)
        self.assertEqual(float(b.balance.balance), 10000.0)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_max_stake_protection(self):
        conn = DerivConnection(app_id="1089")
        original = os.environ.get("DERIV_AUTO_MAX_STAKE")
        try:
            os.environ["DERIV_AUTO_MAX_STAKE"] = "5"
            with self.assertRaises(ValueError):
                await conn.buy_contract(symbol="R_75", contract_type="CALL", amount=10.0, duration=60)
        finally:
            if original is None:
                os.environ.pop("DERIV_AUTO_MAX_STAKE", None)
            else:
                os.environ["DERIV_AUTO_MAX_STAKE"] = original


class TestRiseFallStrategy(unittest.TestCase):
    def test_placeholder(self):
        """À implémenter : tests pour RiseFallStrategy."""
        self.assertTrue(True)
