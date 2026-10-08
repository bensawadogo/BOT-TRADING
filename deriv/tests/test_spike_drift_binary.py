"""La stratégie live doit appliquer EXACTEMENT la règle validée en walk-forward."""
import numpy as np
import pandas as pd
import pytest

from deriv.strategies.spike_drift_binary import (
    SpikeDriftBinaryStrategy,
    closed_candles,
)


def _serie_boom(n=2600, seed=7):
    """Indice type BOOM : dérive baissière + spikes haussiers rares."""
    rng = np.random.default_rng(seed)
    moves = rng.normal(-0.002, 0.01, n)                 # % par bougie
    spikes = rng.random(n) < 0.012
    moves[spikes] += rng.uniform(0.2, 0.8, spikes.sum())
    closes = 1000 * np.cumprod(1 + moves / 100)
    idx = pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC")
    return pd.DataFrame({"close": closes}, index=idx)


def test_signal_identique_a_la_regle_du_walk_forward():
    df = _serie_boom()
    strat = SpikeDriftBinaryStrategy("BOOM500", quantile=0.99, calib_window=1500)
    closes = df["close"].to_numpy()
    n_signaux = 0
    for i in range(1500, len(df)):
        # Règle de référence recalculée à la main (sans le code partagé)
        calib = closes[i - 1500:i]
        thr = max(0.08, np.quantile(np.abs(np.diff(calib) / calib[:-1] * 100), 0.99))
        attendu = (closes[i] - closes[i - 1]) / closes[i - 1] * 100 >= thr
        sig = strat.evaluate(df.iloc[:i + 1])
        assert (sig is not None) == attendu, f"bougie {i}"
        if sig is not None:
            n_signaux += 1
            assert sig.contract_type == "PUT"              # BOOM → Fall
            assert sig.threshold_pct == pytest.approx(thr)
    assert n_signaux >= 5                                   # le test a du sens


def test_aucune_fuite_la_bougie_testee_est_hors_calibration():
    df = _serie_boom()
    strat = SpikeDriftBinaryStrategy("BOOM500", calib_window=1500)
    base = strat.evaluate(df.iloc[:1801])
    strat2 = SpikeDriftBinaryStrategy("BOOM500", calib_window=1500)
    # Modifier une bougie FUTURE ne change rien ; modifier la plus ancienne
    # de la fenêtre (hors calibration) non plus.
    df2 = df.copy()
    df2.iloc[1801:, 0] *= 1.5
    df2.iloc[:300, 0] *= 0.5
    assert (strat2.evaluate(df2.iloc[:1801]) is None) == (base is None)


def test_crash_parie_a_la_hausse():
    df = _serie_boom()
    df["close"] = 2000 - df["close"]                         # miroir : spikes baissiers
    strat = SpikeDriftBinaryStrategy("CRASH500")
    sigs = [strat.evaluate(df.iloc[:i + 1]) for i in range(1500, len(df))]
    sigs = [s for s in sigs if s]
    assert sigs and all(s.contract_type == "CALL" for s in sigs)


def test_une_seule_decision_par_bougie_et_historique_insuffisant():
    df = _serie_boom()
    strat = SpikeDriftBinaryStrategy("BOOM500")
    assert strat.evaluate(df.iloc[:1000]) is None            # < 1501 bougies
    for i in range(1500, len(df)):
        if strat.evaluate(df.iloc[:i + 1]):
            assert strat.evaluate(df.iloc[:i + 1]) is None   # même bougie : ignorée
            break
    assert strat.contract_duration(60) == 600


def test_bougie_en_cours_retiree():
    idx = pd.to_datetime(["2026-01-01 10:00", "2026-01-01 10:01"], utc=True)
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)
    now = pd.Timestamp("2026-01-01 10:01:30", tz="UTC")     # 10:01 pas finie
    assert list(closed_candles(df, 60, now).index) == [idx[0]]
    now = pd.Timestamp("2026-01-01 10:02:00", tz="UTC")
    assert len(closed_candles(df, 60, now)) == 2
