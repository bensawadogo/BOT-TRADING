"""Tests du bot de tendance : signaux, construction, risque, cycle complet, adaptateur MT5."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trend_bot import runner
from trend_bot.brokers.mt5 import MT5Broker
from trend_bot.brokers.paper import PaperBroker
from trend_bot.portfolio import SymbolSpec, needs_trade, target_weights, weight_to_lots
from trend_bot.risk import KillSwitch, apply_regime, regime_series
from trend_bot.signals import daily_returns, ex_ante_vol, per_market, trend_signal


def _prix(n=1400, drift=0.0005, vol=0.01, seed=0, start="2020-01-01"):
    rng = np.random.default_rng(seed)
    r = drift + rng.normal(0, vol, n)
    return pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range(start, periods=n))


# ── signaux ───────────────────────────────────────────────────────────────────
def test_trend_signal_bornes_et_nan_initial():
    s = trend_signal(_prix())
    assert s.iloc[:252].isna().all()
    assert s.dropna().between(-1, 1).all()
    assert trend_signal(_prix(drift=0.003)).iloc[-1] == 1.0


def test_per_market_ignore_les_jours_feries():
    a, b = _prix(seed=1), _prix(seed=2)
    b = b.drop(b.index[-3:-1])                    # b fermé deux jours avant la fin
    df = pd.DataFrame({"a": a, "b": b})
    assert trend_signal(df).isna().iloc[-3:-1]["b"].all()       # piège évité
    s = per_market(trend_signal, df)
    assert s.iloc[-3:].notna().all().all()
    r = daily_returns(df)
    assert np.isclose(r["b"].iloc[-1], b.iloc[-1] / b.iloc[-2] - 1)


# ── construction ──────────────────────────────────────────────────────────────
def test_target_weights_vise_la_vol_du_portefeuille():
    df = pd.DataFrame({f"m{i}": _prix(seed=i, vol=0.005 * (i + 1)) for i in range(4)})
    rets = daily_returns(df)
    sig = pd.Series(1.0, index=df.columns)
    w = target_weights(sig, ex_ante_vol(df).iloc[-1], rets, port_vol=0.10, max_gross=100)
    vol_realisee = (rets.iloc[-252:] @ w).std() * np.sqrt(252)
    assert abs(vol_realisee - 0.10) < 0.01
    assert w["m0"] > w["m3"] > 0                  # moins volatil → plus gros poids
    w2 = target_weights(sig, ex_ante_vol(df).iloc[-1], rets, port_vol=0.10, max_gross=0.5)
    assert np.isclose(w2.abs().sum(), 0.5)


def test_weight_to_lots_et_zone_neutre():
    spec = SymbolSpec(lot_value=100_000, volume_min=0.01, volume_max=50, volume_step=0.01)
    assert weight_to_lots(0.5, 10_000, spec) == 0.05
    assert weight_to_lots(-0.5, 10_000, spec) == -0.05
    assert weight_to_lots(0.0004, 10_000, spec) == 0.0          # sous le minimum
    assert not needs_trade(0.10, 0.11, 0.01)                     # écart < 25 %
    assert needs_trade(0.10, 0.20, 0.01)
    assert needs_trade(0.10, -0.10, 0.01)                        # changement de sens
    assert needs_trade(0.10, 0.0, 0.01)                          # fermeture


# ── risque ────────────────────────────────────────────────────────────────────
def test_killswitch():
    ks = KillSwitch(soft=0.10, hard=0.20)
    assert ks.update(100) == 1.0
    assert ks.update(89) == 0.5
    assert ks.update(79) == 0.0 and ks.tripped
    assert ks.update(120) == 0.0                                 # reste coupé
    ks.reset(120)
    assert ks.update(120) == 1.0


def test_apply_regime():
    assert apply_regime(0.66, -1) == 0.0
    assert apply_regime(0.66, 1) == 0.66
    assert apply_regime(-0.33, 0) == -0.33
    assert apply_regime(0.66, None) == 0.66


def test_regime_series_causal():
    p = _prix(n=900, seed=5)
    plein = regime_series(p, train_len=400, refit_every=50)
    court = regime_series(p.iloc[:700], train_len=400, refit_every=50)
    assert plein.iloc[:700].equals(court) or np.allclose(plein.iloc[:700], court, equal_nan=True)
    assert plein.dropna().isin([-1, 0, 1]).all()


# ── cycle complet (courtier papier) ───────────────────────────────────────────
@pytest.fixture
def papier(tmp_path):
    closes = pd.DataFrame({"UP": _prix(drift=0.002, seed=1), "DOWN": _prix(drift=-0.002, seed=2),
                           "FLAT": _prix(drift=0.0, seed=3)})
    return PaperBroker(closes, tmp_path / "broker.json"), tmp_path


def test_cycle_simulation_ne_trade_pas(papier):
    b, tmp = papier
    lignes = runner.run_cycle(b, True, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN", "FLAT"])
    assert b.positions() == {}
    assert any(l.action != "aucune" for l in lignes)
    assert (tmp / "j.csv").read_text().count("\n") == 4


def test_cycle_reel_puis_zone_neutre(papier):
    b, tmp = papier
    l1 = runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    pos = b.positions()
    assert pos and all(pos.get(l.symbole, 0.0) == pytest.approx(l.lots_cibles) for l in l1)
    lignes = runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    assert all(l.action == "aucune" for l in lignes)             # rien n'a changé


def test_cycle_coupe_circuit_ferme_tout(papier):
    b, tmp = papier
    runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    (tmp / "st.json").write_text('{"peak": 1000000.0, "tripped": false}')
    runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    assert b.positions() == {}


def test_check_account(papier, monkeypatch):
    b, _ = papier
    runner.check_account(b, live=True)                           # démo : OK
    monkeypatch.setattr(b, "is_demo", lambda: False)
    with pytest.raises(RuntimeError):
        runner.check_account(b, live=True)
    runner.check_account(b, live=False)                          # simulation : OK


# ── adaptateur MT5 (faux terminal) ────────────────────────────────────────────
class FauxMT5:
    TIMEFRAME_D1, TRADE_ACTION_DEAL = 16408, 1
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1
    POSITION_TYPE_BUY, POSITION_TYPE_SELL = 0, 1
    ORDER_FILLING_FOK, ORDER_FILLING_IOC, ORDER_FILLING_RETURN = 0, 1, 2
    ORDER_TIME_GTC = 0
    TRADE_RETCODE_DONE, TRADE_RETCODE_PLACED = 10009, 10008
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_MARGIN_MODE_RETAIL_HEDGING = 2

    def __init__(self, hedging=True, demo=True):
        self.sent, self.hedging, self.demo, self.rates_args = [], hedging, demo, None
        self.pos = [SimpleNamespace(ticket=11, symbol="EURUSD", type=0, volume=0.30, magic=770077),
                    SimpleNamespace(ticket=12, symbol="EURUSD", type=0, volume=0.20, magic=770077),
                    SimpleNamespace(ticket=99, symbol="EURUSD", type=0, volume=5.0, magic=1)]

    def initialize(self, **kw): return True
    def last_error(self): return (0, "")
    def shutdown(self): pass
    def symbols_get(self): return [SimpleNamespace(name=n) for n in ("EURUSD", "US Tech 100")]
    def symbol_select(self, n, v): return True
    def account_info(self):
        return SimpleNamespace(equity=10_000.0, trade_mode=0 if self.demo else 2,
                               margin_mode=2 if self.hedging else 0)
    def positions_get(self, symbol=None):
        return [p for p in self.pos if symbol is None or p.symbol == symbol]
    def symbol_info(self, s):
        return SimpleNamespace(trade_tick_value=1.0, trade_tick_size=0.00001, volume_min=0.01,
                               volume_max=100, volume_step=0.01, filling_mode=2)
    def symbol_info_tick(self, s): return SimpleNamespace(bid=1.1000, ask=1.1002)
    def copy_rates_from_pos(self, s, tf, start, n):
        self.rates_args = (s, tf, start, n)
        t = (pd.bdate_range("2024-01-01", periods=n).astype("int64") // 10**9).to_numpy()
        return np.array(list(zip(t, np.linspace(1, 2, n))), dtype=[("time", "i8"), ("close", "f8")])
    def order_send(self, req):
        self.sent.append(req)
        return SimpleNamespace(retcode=10009, volume=req["volume"], price=req["price"], comment="")


def test_mt5_lecture():
    f = FauxMT5()
    b = MT5Broker(f)
    assert b.resolve("NAS100", ["US Tech 100", "USTEC"]) == "US Tech 100"
    assert b.resolve("GER40", ["Germany 40"]) is None
    assert b.positions() == {"EURUSD": 0.5}                       # magic étranger ignoré
    c = b.daily_closes("EURUSD", 300)
    assert f.rates_args[2] == 1 and len(c) == 300                 # bougie en cours exclue
    assert b.is_demo() is True
    assert b.spec("EURUSD").lot_value == pytest.approx(110_010, rel=1e-6)


def test_mt5_hedging_reduit_par_ticket_puis_ouvre():
    f = FauxMT5(hedging=True)
    r = MT5Broker(f).market_order("EURUSD", -0.70)
    assert r.ok and r.lots == pytest.approx(-0.70)
    assert [q.get("position") for q in f.sent] == [11, 12, None]
    assert [q["volume"] for q in f.sent] == [0.30, 0.20, 0.20]
    assert all(q["type"] == f.ORDER_TYPE_SELL and q["magic"] == 770077 for q in f.sent)
    assert f.sent[0]["type_filling"] == f.ORDER_FILLING_IOC


def test_mt5_netting_un_seul_ordre():
    f = FauxMT5(hedging=False)
    r = MT5Broker(f).market_order("EURUSD", 0.25)
    assert r.ok and len(f.sent) == 1 and "position" not in f.sent[0]
    assert f.sent[0]["type"] == f.ORDER_TYPE_BUY and f.sent[0]["price"] == 1.1002


def test_mt5_ordre_refuse():
    f = FauxMT5(hedging=False)
    f.order_send = lambda req: SimpleNamespace(retcode=10019, volume=0, price=0, comment="no money")
    r = MT5Broker(f).market_order("EURUSD", 0.25)
    assert not r.ok and "10019" in r.message
