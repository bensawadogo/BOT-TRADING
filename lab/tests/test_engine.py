"""Le moteur ne doit ni voir le futur ni enjoliver les résultats."""
import numpy as np
import pandas as pd

from lab.engine import ExitRules, Signal, atr14, run


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


def test_atr_positif():
    assert (atr14(_df(np.linspace(100, 110, 50)))[1:] > 0).all()
