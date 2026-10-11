"""Le moteur ne doit ni voir le futur ni enjoliver les résultats."""
import numpy as np
import pandas as pd

from lab.engine import ExitRules, Signal, atr14, run
from lab.strategies import STRATEGIES


def _df(closes, spread=0.5):
    c = np.asarray(closes, float)
    idx = pd.date_range("2026-01-01", periods=len(c), freq="h")
    return pd.DataFrame({"open": c, "high": c + spread, "low": c - spread, "close": c}, index=idx)


def test_entree_a_l_ouverture_suivante_et_cout_deduit():
    df = _df([100, 100, 110, 110, 110])
    r = run(df, {0: Signal(+1)}, ExitRules(max_hold=2), cost=0.001)
    t = r.trades[0]
    assert t.entry_i == 1 and t.entry == 100          # ouverture de la bougie suivante
    assert abs(t.ret - (110 / 100 - 1 - 0.001)) < 1e-12


def test_stop_compte_avant_objectif_dans_la_meme_bougie():
    df = _df([100] * 20)
    df.iloc[16, df.columns.get_loc("high")] = 200        # objectif ET stop touchés
    df.iloc[16, df.columns.get_loc("low")] = 1
    r = run(df, {15: Signal(+1, sl_atr=1, tp_atr=1)}, ExitRules(max_hold=5), cost=0)
    assert r.trades[0].reason == "stop" and r.trades[0].ret < 0


def test_signaux_causaux_le_futur_ne_change_pas_le_passe():
    rng = np.random.default_rng(1)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, 600))
    df = _df(c, spread=0.3)
    reg = pd.DataFrame({"regime": "Bull", "confiance": 0.9}, index=df.index)
    for nom in ("A6", "A5", "C", "D"):
        complet, _ = STRATEGIES[nom](df, regimes=reg, minutes_par_bougie=60)
        tronque, _ = STRATEGIES[nom](df.iloc[:400], regimes=reg.iloc[:400], minutes_par_bougie=60)
        avant = {k: v.side for k, v in complet.items() if k < 400}
        assert avant == {k: v.side for k, v in tronque.items()}, nom


def test_atr_positif():
    assert (atr14(_df(np.linspace(100, 110, 50)))[1:] > 0).all()
