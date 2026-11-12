"""Causalité et exactitude de la couche régime (HMM) et de la couche Fibonacci."""
import numpy as np
import pandas as pd
import pytest

from lab.engine import ExitRules, Signal, run
from lab.fibo import fibo_signals, zigzag
from lab.hmm_causal import (features, fit_baum_welch, forward_filter, viterbi_endpoints,
                            walk_forward_hmm)


def _marche(n=1400, seed=3):
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], size=n // 100 + 1), 100)[:n]
    r = 0.0004 * regime + rng.normal(0, 0.006, n) * (1 + 0.5 * (regime == 0))
    close = pd.Series(100 * np.exp(np.cumsum(r)),
                      index=pd.date_range("2020-01-01", periods=n, freq="D"))
    bruit = np.abs(rng.normal(0, 0.003, n))
    return pd.DataFrame({"open": close.shift(1).fillna(close), "high": close * (1 + bruit),
                         "low": close * (1 - bruit), "close": close})


@pytest.fixture(scope="module")
def modele():
    X = features(_marche()["close"]).dropna().to_numpy()[:600]
    X = (X - X.mean(0)) / X.std(0)
    return fit_baum_welch(X), X


def test_forward_egale_predict_proba_tronque(modele):
    m, X = modele
    p = forward_filter(m, X)
    for t in (10, 250, 599):
        assert np.allclose(p[t], m.predict_proba(X[:t + 1])[-1], atol=1e-6)


def test_viterbi_dernier_etat_egale_decode_tronque(modele):
    m, X = modele
    v = viterbi_endpoints(m, X)
    for t in (10, 250, 599):
        assert v[t] == m.decode(X[:t + 1])[1][-1]


def test_walk_forward_hmm_causal():
    close = _marche()["close"]
    plein = walk_forward_hmm(close, train_len=300, refit_every=100)
    for cut in (500, 777, 1100):
        court = walk_forward_hmm(close.iloc[:cut + 1], train_len=300, refit_every=100)
        assert np.allclose(plein.iloc[cut].to_numpy(float), court.iloc[cut].to_numpy(float),
                           equal_nan=True)
    assert plein["p_bull"].notna().sum() > 0


def test_zigzag_et_fibo_causaux():
    df = _marche()
    sw, s_plein = zigzag(df), fibo_signals(df)
    assert s_plein, "le test doit produire des signaux"
    for cut in (400, 900, 1300):
        sw2 = zigzag(df.iloc[:cut + 1])
        assert np.allclose(sw.h1[:cut + 1], sw2.h1, equal_nan=True)
        assert np.allclose(sw.l1[:cut + 1], sw2.l1, equal_nan=True)
        s_court = fibo_signals(df.iloc[:cut + 1])
        assert {i for i in s_plein if i <= cut} == set(s_court)


def test_fibo_niveaux_coherents():
    df = _marche()
    for s in fibo_signals(df).values():
        assert s.side * (s.tp_price - s.sl_price) > 0


def test_moteur_niveaux_absolus():
    df = pd.DataFrame({"open": [100, 100, 101, 103], "high": [100, 101, 104, 104],
                       "low": [100, 99.5, 100.5, 102], "close": [100, 101, 103, 103]})
    res = run(df, {0: Signal(1, sl_price=99.0, tp_price=103.5)}, ExitRules(max_hold=5), 0.0)
    assert res.trades[0].reason == "objectif" and res.trades[0].exit == 103.5
    # ouverture déjà sous le stop : trade ignoré
    res = run(df, {0: Signal(1, sl_price=100.5, tp_price=103.5)}, ExitRules(max_hold=5), 0.0)
    assert res.trades == []
