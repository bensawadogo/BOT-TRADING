"""Stratégie Freqtrade guidée par le régime HMM partagé."""

from __future__ import annotations

from pandas import DataFrame

try:
    from freqtrade.strategy import IStrategy
except ImportError:  # permet l'import hors Freqtrade

    class IStrategy:  # type: ignore[no-redef]
        INTERFACE_VERSION = 3
        timeframe = "5m"
        can_short = False
        minimal_roi = {"0": 0.02}
        stoploss = -0.03
        startup_candle_count = 200

        def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            return dataframe

        def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            return dataframe

        def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            return dataframe


class HMMRegimeStrategy(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    minimal_roi = {"0": 0.03, "30": 0.015, "60": 0.005}
    stoploss = -0.04
    startup_candle_count = 200

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from intelligence.kalman_filter import KalmanFilter1D
        from intelligence.hmm_regime import HMMRegimeDetector, Regime

        prices = dataframe["close"]
        kf = KalmanFilter1D().filter(prices)
        dataframe["kalman_price"] = kf["kalman_price"]
        dataframe["kalman_velocity"] = kf["kalman_velocity"]
        try:
            hmm = HMMRegimeDetector()
            hmm.fit(prices)
            regimes = hmm.predict(prices)
            dataframe["hmm_regime"] = regimes.reindex(dataframe.index).ffill().fillna(int(Regime.RANGE))
        except Exception:
            dataframe["hmm_regime"] = int(Regime.RANGE)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from intelligence.hmm_regime import Regime

        dataframe.loc[
            (dataframe["hmm_regime"] == int(Regime.BULL))
            & (dataframe["kalman_velocity"] > 0)
            & (dataframe["close"] > dataframe["kalman_price"]),
            "enter_long",
        ] = 1
        dataframe.loc[
            (dataframe["hmm_regime"] == int(Regime.BEAR))
            & (dataframe["kalman_velocity"] < 0)
            & (dataframe["close"] < dataframe["kalman_price"]),
            "enter_short",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from intelligence.hmm_regime import Regime

        dataframe.loc[dataframe["hmm_regime"] == int(Regime.RANGE), "exit_long"] = 1
        dataframe.loc[dataframe["hmm_regime"] == int(Regime.RANGE), "exit_short"] = 1
        return dataframe
