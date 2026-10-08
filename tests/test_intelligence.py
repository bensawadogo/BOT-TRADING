"""Tests unitaires et de validation pour la couche intelligence partagée."""

import os
import sys
import unittest
import numpy as np
import pandas as pd

# S'assurer que la racine du projet est dans sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from intelligence.hmm_regime import HMMRegimeDetector, Regime
from intelligence.kalman_filter import KalmanPriceFilter, KalmanFilter1D
from intelligence.signal_engine import SignalEngine, TradingSignal


def generate_synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Génère un DataFrame OHLCV synthétique réaliste."""
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


class TestHMMRegimeDetector(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_ohlcv(250)
        self.detector = HMMRegimeDetector()

    def test_features_shape(self):
        feats = self.detector.prepare_features(self.df)
        # Log return et volatilité 20 périodes entraînent un drop des 20 premières lignes
        self.assertEqual(feats.shape[1], 3)
        self.assertGreater(feats.shape[0], 200)

    def test_features_without_volume(self):
        df_no_vol = self.df[["close"]].copy()
        feats = self.detector.prepare_features(df_no_vol)
        self.assertEqual(feats.shape[1], 3)

    def test_train_and_predict(self):
        train_res = self.detector.train(self.df)
        self.assertIn("regime_labels", train_res)
        self.assertIn("transition_matrix", train_res)
        self.assertEqual(len(train_res["regime_labels"]), 3)

        pred = self.detector.predict(self.df)
        self.assertIn("regime", pred)
        self.assertIn("confiance", pred)
        self.assertIn("probas", pred)
        self.assertIn(pred["regime"], ["Bull", "Bear", "Range"])
        self.assertGreaterEqual(pred["confiance"], 0.0)
        self.assertLessEqual(pred["confiance"], 1.0)
        # Somme des probabilités proche de 1
        self.assertAlmostEqual(sum(pred["probas"].values()), 1.0, delta=0.05)

    def test_backward_compatibility(self):
        self.detector.train(self.df)
        reg = self.detector.current_regime(self.df["close"])
        self.assertIn(reg, [Regime.BULL, Regime.BEAR, Regime.RANGE])
        series_pred = self.detector.predict(self.df["close"])
        self.assertIsInstance(series_pred, pd.Series)


class TestKalmanPriceFilter(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_ohlcv(100)
        self.kf = KalmanPriceFilter()

    def test_update(self):
        p1 = self.kf.update(100.0)
        self.assertEqual(p1, 100.0)
        p2 = self.kf.update(105.0)
        self.assertGreater(p2, 100.0)
        self.assertLess(p2, 105.0)

    def test_apply_to_series(self):
        s = self.kf.apply_to_series(self.df["close"])
        self.assertEqual(len(s), len(self.df))
        self.assertEqual(s.name, "kalman_price")

    def test_signals(self):
        self.kf.x = 100.0
        self.assertEqual(self.kf.signal(102.0, threshold=0.01), "SELL")
        self.assertEqual(self.kf.signal(98.0, threshold=0.01), "BUY")
        self.assertEqual(self.kf.signal(100.5, threshold=0.01), "NEUTRAL")

    def test_compatibility_filter(self):
        df_out = self.kf.filter(self.df["close"])
        self.assertIn("kalman_price", df_out.columns)
        self.assertIn("kalman_velocity", df_out.columns)
        slope = self.kf.slope_sign(self.df["close"])
        self.assertIn(slope, [-1, 0, 1])


class TestSignalEngine(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_ohlcv(250)
        self.engine = SignalEngine()
        # Entraînement préalable du HMM
        self.engine.hmm.train(self.df)

    def test_calculate_score(self):
        res = self.engine.calculate_score(self.df)
        self.assertIn("score", res)
        self.assertIn("signal", res)
        self.assertIn("details", res)
        self.assertIn("force", res)
        self.assertGreaterEqual(res["score"], 0.0)
        self.assertLessEqual(res["score"], 100.0)
        self.assertIn(res["signal"], ["BUY", "SELL", "HOLD"])
        self.assertIn(res["force"], ["FORT", "MOYEN", "FAIBLE"])

    def test_insufficient_data(self):
        small_df = self.df.iloc[:20]
        res = self.engine.calculate_score(small_df)
        self.assertEqual(res["signal"], "HOLD")
        self.assertEqual(res["score"], 50.0)

    def test_message_telegram(self):
        res = self.engine.calculate_score(self.df)
        msg = self.engine.message_telegram("BTC/USDT", res)
        self.assertIn("BTC/USDT", msg)
        self.assertIn("Score :", msg)
        self.assertIn("Régime HMM", msg)
        self.assertIn("Kalman", msg)
        self.assertIn("RSI", msg)
        self.assertIn("MACD", msg)
        self.assertIn("Bollinger", msg)

    def test_generate_compatibility(self):
        sig = self.engine.generate(self.df["close"])
        self.assertIsInstance(sig, TradingSignal)
        self.assertIn(sig.direction, [-1, 0, 1])
        self.assertGreaterEqual(sig.confidence, 0.0)
        self.assertLessEqual(sig.confidence, 1.0)


class TestStrategiesIntegration(unittest.TestCase):
    def setUp(self):
        self.df = generate_synthetic_ohlcv(250)

    def test_deriv_rise_fall_strategy(self):
        from deriv.strategies.rise_fall import RiseFallStrategy
        strat = RiseFallStrategy()
        strat.engine.hmm.train(self.df)
        sig = strat.evaluate(self.df["close"])
        # Peut être None si neutre ou TradingSignal si actif
        if sig is not None:
            prop = strat.to_proposal(sig, stake=1.0)
            self.assertIn("contract_type", prop)

    def test_crypto_hmm_regime_strategy(self):
        from crypto.strategies.HMMRegimeStrategy import HMMRegimeStrategy
        strat = HMMRegimeStrategy()
        ind_df = strat.populate_indicators(self.df.copy(), {})
        self.assertIn("kalman_price", ind_df.columns)
        self.assertIn("hmm_regime", ind_df.columns)
        trend_df = strat.populate_entry_trend(ind_df, {})
        self.assertIn("enter_long", trend_df.columns)


if __name__ == "__main__":
    unittest.main()
