"""Tests unitaires du bot Deriv ensemble 4/5 (collecteur, modèles, vote, risque)."""

import asyncio
import os
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deriv.data_collector import DerivDataCollector
from deriv.ensemble_predictor import (
    EnsemblePredictor,
    HMMPredictor,
    KalmanMomentumPredictor,
    RSIDivergencePredictor,
    build_features,
    _TF_OK,
)
from deriv.risk_manager import RiskManager, TradeResult


def generate_synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique réaliste (même graine que tests existants)."""
    rng = np.random.default_rng(42)
    dt_index = pd.date_range("2025-01-01", periods=n, freq="5min")
    returns = rng.normal(0.0002, 0.005, size=n)
    close_prices = 100.0 * np.exp(np.cumsum(returns))
    high = close_prices * (1.0 + rng.uniform(0.001, 0.004, size=n))
    low = close_prices * (1.0 - rng.uniform(0.001, 0.004, size=n))
    open_prices = low + rng.uniform(0.1, 0.9, size=n) * (high - low)
    volume = rng.uniform(100.0, 500.0, size=n)
    return pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
    }, index=dt_index)


class TestDerivDataCollector(unittest.TestCase):
    """Vérifie l'agrégation tick → bougie OHLCV (sans API)."""

    def test_build_candles_from_ticks(self):
        col = DerivDataCollector("frxEURUSD", tick_size=60)
        # 3600 ticks de 1s → ~60 bougies de 60s
        for epoch in range(1, 3601):
            col.process_tick(epoch, 100.0 + 0.001 * epoch)
        self.assertGreaterEqual(len(col.candles), 60)

        df = col.get_dataframe()
        self.assertFalse(df.empty)
        self.assertGreaterEqual(len(df), 50)
        self.assertTrue({"open", "high", "low", "close", "volume"}
                        .issubset(df.columns))

        # Cohérence OHLC
        for _, row in df.iterrows():
            self.assertLessEqual(row["low"], row["open"])
            self.assertGreaterEqual(row["high"], row["close"])

    def test_dataframe_vide_avant_50_bougies(self):
        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 200):
            col.process_tick(epoch, 100.0)
        self.assertTrue(col.get_dataframe().empty)

    def test_symbols_definis(self):
        self.assertIn("frxEURUSD", DerivDataCollector.SYMBOLES)
        self.assertIn("R_75", DerivDataCollector.SYMBOLES)


class TestFeatures(unittest.TestCase):
    def test_build_features_shape_and_target(self):
        df = generate_synthetic_ohlcv(300)
        feat = build_features(df)
        self.assertFalse(feat.empty)

        # Anti-fuite : target découplé dans build_labels()
        from deriv.ensemble_predictor import build_labels
        y = build_labels(df)
        self.assertIsNotNone(y)
        self.assertTrue(set(y.dropna().unique()).issubset({0, 1}))

        from deriv.ensemble_predictor import FEATURE_COLS
        self.assertEqual(len(FEATURE_COLS), 11)
        self.assertTrue(all(c in feat.columns for c in FEATURE_COLS))
        self.assertEqual(feat[FEATURE_COLS].isna().sum().sum(), 0)


class TestHMMPredictor(unittest.TestCase):
    def test_train_and_predict(self):
        df = generate_synthetic_ohlcv(300)
        hmm = HMMPredictor()
        hmm.train(df)
        self.assertTrue(hmm.is_trained)

        res = hmm.predict(df)
        for key in ("regime", "viterbi", "confiance", "coherent", "vote"):
            self.assertIn(key, res)
        self.assertIn(res["regime"], ["Bull", "Bear", "Range"])
        self.assertIn(res["vote"], [0, 0.5, 1])

    def test_predict_without_model_neutral(self):
        hmm = HMMPredictor()
        hmm.model = None
        hmm.is_trained = False
        res = hmm.predict(generate_synthetic_ohlcv(100))
        self.assertEqual(res["vote"], 0.5)
        self.assertEqual(res["regime"], "Range")


class TestKalmanMomentumPredictor(unittest.TestCase):
    def test_bullish_vote(self):
        df = generate_synthetic_ohlcv(100)
        df["close"] = 100 + np.arange(100.0) * 0.1  # tendance haussière
        res = KalmanMomentumPredictor().predict(df)
        self.assertEqual(res["vote"], 1)

    def test_bearish_vote(self):
        df = generate_synthetic_ohlcv(100)
        df["close"] = 110 - np.arange(100.0) * 0.1  # tendance baissière
        res = KalmanMomentumPredictor().predict(df)
        self.assertEqual(res["vote"], 0)

    def test_short_data_neutral(self):
        df = generate_synthetic_ohlcv(5)
        res = KalmanMomentumPredictor().predict(df)
        self.assertEqual(res["vote"], 0.5)


class TestRSIDivergencePredictor(unittest.TestCase):
    def test_oversold_vote_buy(self):
        df = generate_synthetic_ohlcv(80)
        df["close"] = 100 - np.arange(80.0) * 0.2  # chute → RSI < 30
        res = RSIDivergencePredictor().predict(df)
        # ADX-filtered : zone survendue sans trend baissier → 0.6 (biais acheteur)
        self.assertIn(res["vote"], [1, 0.6, 0.5])
        if res["vote"] in (1, 0.6):
            self.assertTrue(res["rsi"] < 30 or res["divergence_bull"])

    def test_short_data_neutral(self):
        df = generate_synthetic_ohlcv(5)
        res = RSIDivergencePredictor().predict(df)
        self.assertEqual(res["vote"], 0.5)


class _StubPredictor:
    """Fausse prédiction pour tester la règle de vote 4/5 en isolation."""

    def __init__(self, vote):
        self.vote = vote

    def predict(self, df):
        proba = 0.75 if self.vote == 1 else (0.25 if self.vote == 0 else 0.5)
        return {
            "regime": "Range", "viterbi": "Range", "confiance": 0.0,
            "coherent": True, "vote": self.vote,
            "proba_up": proba, "trend": 0.0, "rsi": 50.0,
            "ema_align": "MIXED", "adx": 15.0, "roc": 0.0,
        }


def make_ensemble(votes: list, trend_vote: float = 0.5, momentum_vote: float = 0.5) -> EnsemblePredictor:
    ens = EnsemblePredictor.__new__(EnsemblePredictor)  # pas de constr (modèles lents)
    ens.hmm, ens.xgb, ens.lstm, ens.kalman, ens.rsi = [
        _StubPredictor(v) for v in votes
    ]
    ens.trend = _StubPredictor(trend_vote)
    ens.momentum = _StubPredictor(momentum_vote)
    return ens


class TestEnsembleVotingRule(unittest.TestCase):
    """La RÈGLE ABSOLUE : >= 4/5 pour trader, sinon HOLD."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(120)

    def test_4_buy_1_hold_tradeable(self):
        # 5 votes principaux + trend (0.5) + momentum (0.5) = 7 modèles
        res = make_ensemble([1, 1, 1, 1, 0.5]).predict(self.df)
        self.assertEqual(res["signal"], "BUY")
        self.assertTrue(res["tradeable"])
        self.assertEqual(res["buy_votes"], 4)
        self.assertEqual(res["hold_votes"], 3)
        self.assertAlmostEqual(res["confiance"], 4/7, places=2)

    def test_4_sell_tradeable(self):
        res = make_ensemble([0, 0, 0, 0, 1]).predict(self.df)
        self.assertEqual(res["signal"], "SELL")
        self.assertTrue(res["tradeable"])
        self.assertEqual(res["sell_votes"], 4)

    def test_3_buy_2_sell_hold(self):
        res = make_ensemble([1, 1, 1, 0, 0]).predict(self.df)
        self.assertEqual(res["signal"], "HOLD")
        self.assertFalse(res["tradeable"])
        self.assertEqual(res["confiance"], 0.0)

    def test_all_neutral_hold(self):
        res = make_ensemble([0.5, 0.5, 0.5, 0.5, 0.5]).predict(self.df)
        self.assertEqual(res["signal"], "HOLD")
        self.assertFalse(res["tradeable"])

    @unittest.skipUnless(not _TF_OK, "Test LSTM fallback quand tensorflow absent")
    def test_lstm_fallback_neutral(self):
        from deriv.ensemble_predictor import LSTMPredictor
        lstm = LSTMPredictor()
        self.assertFalse(lstm.is_trained)
        res = lstm.predict(self.df)
        self.assertEqual(res["vote"], 0.5)

    def test_regle_absolue_constante(self):
        self.assertEqual(EnsemblePredictor.MIN_VOTES_TO_TRADE, 4)


class TestRiskManagerCompat(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(capital=150.0)

    def test_check_stake_ok(self):
        ok, _ = self.rm.check(2.0)
        self.assertTrue(ok)
        # 2$ <= min(2% de 150 = 3$, auto_max 5) → auto OK
        self.assertEqual(self.rm.max_stake(), 3.0)

    def test_check_stake_above_limit_confirmation(self):
        ok, raison = self.rm.check(20.0)
        self.assertFalse(ok)
        self.assertIn("CONFIRMATION_REQUISE", raison)

    def test_record_traderesult(self):
        self.rm.record(TradeResult(symbol="frxEURUSD", pnl=-2.0,
                                   contract_type="CALL", stake=2.0))
        self.assertEqual(self.rm.total_trades, 1)
        self.assertEqual(self.rm.capital_actuel, 148.0)

    def test_stats_keys(self):
        stats = self.rm.stats
        for key in ("winrate_jour", "total_trades", "winning_trades",
                    "daily_loss", "consecutive_loss", "capital_actuel"):
            self.assertIn(key, stats)
        self.assertEqual(stats["winrate_jour"], 0.0)

    def test_stop_apres_3_pertes_consecutives(self):
        for _ in range(3):
            self.rm.record(-2.0)
        ok, raison = self.rm.can_trade()
        self.assertFalse(ok)
        self.assertIn("3 pertes consécutives", raison)


class TestBotExecutorGate(unittest.TestCase):
    """Le portail d'exécution ne doit JAMAIS trader sur HOLD."""

    def setUp(self):
        from deriv.bot_executor import DerivBotExecutor
        self.bot = DerivBotExecutor(capital_usd=150.0)
        self.df = generate_synthetic_ohlcv(120)

    def test_instanciation(self):
        self.assertIsNone(self.bot.conn)
        self.assertFalse(self.bot._telegram_active)  # placeholder TON_TOKEN
        self.assertEqual(self.bot.STAKE_AUTO, 2.0)

    def test_telegram_desactive_sans_token(self):
        # Pas d'erreur sans token réel (aiohttp jamais contacté)
        asyncio.run(self.bot._telegram("test"))
        self.assertIsNone(self.bot.conn)

    def test_run_cycle_hold_ne_trade_pas(self):
        self.bot.ensemble = make_ensemble([0.5, 0.5, 0.5, 0.5, 0.5])
        asyncio.run(self.bot._run_cycle(self.df))
        self.assertIsNone(self.bot.conn)

    def test_run_cycle_3_votes_hold_ne_trade_pas(self):
        # 3 BUY / 2 SELL → HOLD (règle 4/5) → aucune exécution
        self.bot.ensemble = make_ensemble([1, 1, 1, 0, 0])
        asyncio.run(self.bot._run_cycle(self.df))
        self.assertIsNone(self.bot.conn)


if __name__ == "__main__":
    unittest.main()