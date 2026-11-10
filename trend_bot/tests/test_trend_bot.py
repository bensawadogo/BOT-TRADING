"""Tests du bot de tendance : signaux, construction, risque, cycle complet, adaptateur MT5."""
from functools import partial
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from trend_bot import runner
from trend_bot.backtest import Options, backtest
from trend_bot.brokers.mt5 import MT5Broker
from trend_bot.brokers.paper import PaperBroker
from trend_bot.portfolio import SymbolSpec, needs_trade, target_weights, weight_to_lots
from trend_bot.risk import KillSwitch, apply_regime, regime_now, regime_series
from trend_bot.signals import (Strategie, daily_returns, ewmac, ex_ante_vol, market_signal,
                               per_market, trend_forecast, trend_signal, weekdays_only)


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


def test_weekdays_only_retire_samedi_et_dimanche():
    p = _prix(n=30).reindex(pd.date_range("2020-01-01", periods=42, freq="D")).ffill()
    assert set(weekdays_only(p).index.day_of_week) == {0, 1, 2, 3, 4}
    assert len(weekdays_only(pd.DataFrame({"a": p}))) == 30


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


def test_ewmac_bornes_nan_initial_et_sens():
    f = ewmac(_prix(drift=0.003), 16, 64, 3.75)
    assert f.iloc[:64].isna().all()
    assert f.dropna().between(-1, 1).all()
    assert f.iloc[-1] > 0.5 and ewmac(_prix(drift=-0.003), 16, 64, 3.75).iloc[-1] < -0.5


def test_trend_forecast_ne_voit_pas_le_futur():
    p = _prix(n=1500, seed=5)
    long, court = trend_forecast(p, desaccord=0.5), trend_forecast(p.iloc[:1200], desaccord=0.5)
    assert np.allclose(long.iloc[:1200].dropna(), court.dropna(), atol=1e-12)


def test_desaccord_reduit_la_prevision_quand_lent_et_rapide_se_contredisent():
    # longue hausse puis chute brutale : la vitesse rapide devient négative avant la lente
    p = pd.Series(np.r_[np.linspace(100, 300, 700), np.linspace(300, 240, 60)],
                  index=pd.bdate_range("2018-01-01", periods=760))
    plein, reduit = trend_forecast(p), trend_forecast(p, desaccord=0.5)
    rapide, lente = ewmac(p, 16, 64, 3.75), ewmac(p, 64, 256, 1.87)
    contredit = (np.sign(rapide) * np.sign(lente) < 0) & plein.notna()
    assert contredit.any()
    assert np.allclose(reduit[contredit], plein[contredit] * 0.5)
    assert np.allclose(reduit[~contredit].dropna(), plein[~contredit].dropna())


def test_market_signal_choisit_la_famille():
    p = _prix()
    assert market_signal(p).equals(trend_signal(p))
    assert market_signal(p, Strategie("ewmac")).equals(trend_forecast(p))


def test_apply_regime_facteur():
    assert apply_regime(0.6, -1, 0.5) == pytest.approx(0.3)
    assert apply_regime(0.6, 1, 0.5) == 0.6


def test_sharpe_degonfle_et_proba_meilleur():
    from trend_bot.backtest import proba_meilleur, sharpe_degonfle
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    bon = pd.Series(rng.normal(0.0015, 0.01, 3000), index=idx)
    bruit = pd.Series(rng.normal(0.0, 0.01, 3000), index=idx)
    sr = [bon.mean() / bon.std(), bruit.mean() / bruit.std()]
    assert sharpe_degonfle(bon, sr) > 0.95 > sharpe_degonfle(bruit, sr)
    jumeau = pd.Series(rng.normal(0.0015, 0.01, 3000), index=idx)
    assert proba_meilleur(bon, bruit) > 0.9
    assert proba_meilleur(bon, jumeau) + proba_meilleur(jumeau, bon) == pytest.approx(1, abs=0.01)


def test_apply_regime():
    assert apply_regime(0.66, -1) == 0.0
    assert apply_regime(0.66, 1) == 0.66
    assert apply_regime(-0.33, 0) == -0.33
    assert apply_regime(0.66, None) == 0.66


def test_regime_series_causal():
    p = _prix(n=900, seed=5)
    plein = regime_series(p, train_len=400)
    court = regime_series(p.iloc[:700], train_len=400)
    assert np.allclose(plein.iloc[:700], court, equal_nan=True)
    assert plein.dropna().isin([-1, 0, 1]).all()


def test_regime_now_egale_le_backtest():
    p = _prix(n=700, seed=6)
    serie = regime_series(p, train_len=300)
    for t in (450, 451, 520, 699):                 # début de mois, milieu, fin
        historique = p.iloc[t - 420:t + 1]       # le live ne voit qu'une fenêtre récente
        assert regime_now(historique, train_len=300) == serie.iloc[t]
    assert regime_now(p.iloc[:200], train_len=300) is None


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


def test_live_egale_backtest(tmp_path, monkeypatch):
    """Le cycle live et le backtest prennent les mêmes positions, avec les bougies de Deriv
    MT5 : forex avec une bougie le dimanche, crypto 7 j/7."""
    jours = pd.date_range("2020-01-01", periods=1500, freq="D")
    fx = _prix(n=1500, seed=11, start="2020-01-01").set_axis(jours)
    fx = fx[jours.day_of_week != 5]                                   # pas de samedi
    crypto = _prix(n=1500, drift=0.001, vol=0.03, seed=12).set_axis(jours)
    indice = _prix(n=1500, drift=-0.0005, seed=13).set_axis(jours)
    indice = indice[jours.day_of_week < 5]
    closes = pd.DataFrame({"FX": fx, "CRYPTO": crypto, "IDX": indice})
    reg = partial(regime_series, train_len=300)
    regimes = pd.DataFrame({c: reg(weekdays_only(closes[c].dropna())) for c in closes}
                           ).reindex(weekdays_only(closes).index).ffill(limit=10)
    res = backtest(closes, {}, Options(killswitch=False), regimes=regimes)
    monkeypatch.setattr(runner, "regime_now", partial(regime_now, train_len=300))
    for t in res.targets.index[[400, 401, 650, -1]]:
        b = PaperBroker(closes.loc[:t])
        lignes = runner.run_cycle(b, True, tmp_path / f"{t:%F}.json", tmp_path / "j.csv",
                                  list(closes))
        live = pd.Series({l.marche: l.poids for l in lignes})
        assert np.allclose(live, res.targets.loc[t, live.index], atol=1e-4), t


def test_cycle_signale_le_lot_minimum(papier, monkeypatch):
    b, tmp = papier
    monkeypatch.setattr(b, "spec", lambda s: SymbolSpec(1e7, 0.01, 100, 0.01))
    lignes = runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    assert b.positions() == {}
    assert all("sous le lot minimum" in l.resultat for l in lignes if l.poids)


def test_cycle_ferme_une_position_hors_panier(papier):
    b, tmp = papier
    b.market_order("FLAT", 3.0)                   # position du bot sur un marché retiré
    lignes = runner.run_cycle(b, False, tmp / "st.json", tmp / "j.csv", ["UP", "DOWN"])
    assert "FLAT" not in b.positions()
    assert any("hors panier" in l.action for l in lignes)


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
        self.contrat = 100_000
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
                               margin_mode=2 if self.hedging else 0, currency="USD")
    def positions_get(self, symbol=None):
        return [p for p in self.pos if symbol is None or p.symbol == symbol]
    def symbol_info(self, s):
        return SimpleNamespace(trade_tick_value=1.0, trade_tick_size=0.00001, volume_min=0.01,
                               volume_max=100, volume_step=0.01, filling_mode=2,
                               trade_contract_size=self.contrat, currency_profit="USD")
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


def test_mt5_specification_incoherente_ne_surdimensionne_pas():
    f = FauxMT5()
    f.contrat = 10_000_000          # contrat 100× plus gros que ce que dit la valeur du tick
    assert MT5Broker(f).spec("EURUSD").lot_value == pytest.approx(11_001_000, rel=1e-6)


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


# ── journal de décision ───────────────────────────────────────────────────────
def test_raison_explique_chaque_cas():
    nan = float("nan")
    assert "historique insuffisant" in runner.raison(nan, None, nan, nan, 0, 1, 0, 0, False)
    t = runner.raison(0.66, -1, 0.0, 0.08, 0.0, 1, 0, 0.1, True)
    assert "tendance +0.66" in t and "baissier contredit" in t and "(coupé)" in t
    t = runner.raison(0.66, 1, 0.66, 0.08, 0.3, 0.5, 0.3, 0.3, False)
    assert "pas de contradiction" in t and "exposition ×0.5" in t and "déjà conforme" in t
    assert "zone neutre" in runner.raison(0.66, 1, 0.66, 0.08, 0.3, 1, 0.3, 0.2, False)
    assert "coupe-circuit : tout est coupé" in runner.raison(0.5, 1, 0.5, 0.08, 0, 0, 0, 0.2, True)


def test_cycle_ecrit_la_raison_de_chaque_decision(papier):
    b, tmp = papier
    lignes = runner.run_cycle(b, True, tmp / "s.json", tmp / "j.csv", ["UP", "DOWN", "FLAT"])
    assert all(l.raison for l in lignes)
    j = pd.read_csv(tmp / "j.csv")
    assert {"signal_brut", "exposition", "raison"} <= set(j.columns)


def test_journal_ancien_format_est_archive_pas_corrompu(papier):
    b, tmp = papier
    (tmp / "j.csv").write_text("date,marche\n2026-01-01,EURUSD\n", encoding="utf-8")
    runner.run_cycle(b, True, tmp / "s.json", tmp / "j.csv", ["UP", "DOWN", "FLAT"])
    assert (tmp / "j_ancien_format.csv").exists()
    assert "raison" in pd.read_csv(tmp / "j.csv").columns


def test_frais_admin_dus_seulement_apres_le_delai_de_grace():
    closes = pd.DataFrame({"A": _prix(n=1500, drift=0.001, seed=21)})
    opt = Options(regime=False, killswitch=False)
    taux = {"A": (0.05, 0.05)}
    base = backtest(closes, {}, opt).equity.iloc[-1]
    sans_delai = backtest(closes, {}, opt, admin=taux, grace=0).equity.iloc[-1]
    avec_delai = backtest(closes, {}, opt, admin=taux, grace=15).equity.iloc[-1]
    jamais = backtest(closes, {}, opt, admin=taux, grace=100000).equity.iloc[-1]
    assert jamais == pytest.approx(base)
    assert sans_delai < avec_delai < base


# ── annonces économiques et garde-fous d'exécution ────────────────────────────
from datetime import datetime, timezone

from trend_bot import news


def _annonce(heure, devise="USD", impact="High", titre="Non-Farm Employment Change"):
    return {"title": titre, "country": devise, "impact": impact,
            "date": f"2026-11-06T{heure}:00+00:00"}


def test_devises_des_marches():
    assert news.devises("EURUSD") == ["EUR", "USD"]
    assert news.devises("NAS100") == ["USD"] and news.devises("GER40") == ["EUR"]
    assert news.devises("INCONNU") == []


def test_annonce_bloque_seulement_dans_la_fenetre_et_sur_la_bonne_devise():
    ev = [_annonce("12:30"), _annonce("12:30", "JPY"), _annonce("12:30", impact="Low")]
    t = lambda h, m: datetime(2026, 11, 6, h, m, tzinfo=timezone.utc)   # noqa: E731
    assert news.bloquantes("EURUSD", t(12, 10), ev)           # 20 min avant
    assert news.bloquantes("EURUSD", t(12, 55), ev)           # 25 min après
    assert not news.bloquantes("EURUSD", t(11, 55), ev)       # 35 min avant
    assert not news.bloquantes("EURUSD", t(13, 10), ev)       # 40 min après
    assert not news.bloquantes("USDJPY", t(12, 30), [_annonce("12:30", "EUR")])
    assert len(news.bloquantes("USDJPY", t(12, 30), ev)) == 2  # USD et JPY, pas l'impact faible


def test_calendrier_injoignable_ne_bloque_pas_et_garde_l_ancien_releve(tmp_path):
    def panne():
        raise OSError("réseau coupé")
    cache = tmp_path / "news.json"
    assert news.charger(cache, fetch=panne) == []                       # rien en cache : pas de filtre
    cache.write_text('[{"country": "USD"}]')
    assert news.charger(cache, fetch=panne, maintenant=cache.stat().st_mtime + 7 * 3600) == [
        {"country": "USD"}]                                             # vieux relevé gardé
    appels = []
    assert news.charger(cache, fetch=lambda: appels.append(1) or [], maintenant=cache.stat().st_mtime + 60)
    assert not appels                                                   # relevé frais : pas de réseau


def test_cycle_reporte_l_ouverture_mais_jamais_la_reduction(papier, monkeypatch):
    b, tmp = papier
    monkeypatch.setattr(runner.config, "NEWS_FILTER", True)
    monkeypatch.setattr(runner.news, "charger", lambda *_: [
        _annonce("12:30", d) for d in ("USD", "EUR", "GBP", "JPY")])
    monkeypatch.setattr(runner.news, "devises", lambda m: ["USD"])
    t = datetime(2026, 11, 6, 12, 20, tzinfo=timezone.utc)
    lignes = runner.run_cycle(b, False, tmp / "s.json", tmp / "j.csv", ["UP", "DOWN", "FLAT"], t)
    ouverts = [l for l in lignes if l.lots_cibles]
    assert ouverts and all(l.action == "reporté" and "annonce" in l.resultat for l in ouverts)
    assert all("reporté" in l.raison for l in ouverts) and not b.positions()
    # hors fenêtre : les mêmes ordres partent
    t2 = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)
    lignes = runner.run_cycle(b, False, tmp / "s2.json", tmp / "j2.csv", ["UP", "DOWN", "FLAT"], t2)
    assert b.positions() and not any(l.action == "reporté" for l in lignes)
    # position existante, cible nulle (réduction) pendant une annonce : jamais retardée
    b.pos = {"UP": 5.0}
    monkeypatch.setattr(runner, "weight_to_lots", lambda *a, **k: 0.0)
    lignes = runner.run_cycle(b, False, tmp / "s3.json", tmp / "j3.csv", ["UP"], t)
    assert lignes[0].action.startswith("vente") and not b.positions()
    # retournement (position vendeuse, cible acheteuse de même taille) : expose à nouveau → reporté
    monkeypatch.undo()
    monkeypatch.setattr(runner.config, "NEWS_FILTER", True)
    monkeypatch.setattr(runner.news, "charger", lambda *_: [_annonce("12:30", "USD")])
    monkeypatch.setattr(runner.news, "devises", lambda m: ["USD"])
    b4, tmp4 = b, tmp
    lignes = runner.run_cycle(b4, True, tmp4 / "s4.json", tmp4 / "j4.csv", ["UP"], t2)
    cible = lignes[0].lots_cibles
    assert cible > 0
    b4.pos = {"UP": -cible}
    lignes = runner.run_cycle(b4, False, tmp4 / "s5.json", tmp4 / "j5.csv", ["UP"], t)
    assert lignes[0].action == "reporté" and b4.positions() == {"UP": -cible}


def test_empecher_la_veille_ne_plante_pas():
    assert runner.empecher_la_veille() in (True, False)


def test_cycle_refuse_d_envoyer_des_ordres_si_algo_trading_est_coupe(papier):
    b, tmp = papier
    b.can_trade = lambda: False
    with pytest.raises(RuntimeError, match="Algo Trading"):
        runner.run_cycle(b, False, tmp / "s.json", tmp / "j.csv", ["UP"])
    runner.run_cycle(b, True, tmp / "s.json", tmp / "j.csv", ["UP"])   # la simulation reste permise


def test_mt5_can_trade_exige_terminal_et_compte():
    ok = SimpleNamespace(trade_allowed=True)
    ko = SimpleNamespace(trade_allowed=False)
    for terminal, compte, attendu in [(ok, ok, True), (ko, ok, False), (ok, ko, False),
                                      (None, ok, False)]:
        b = MT5Broker.__new__(MT5Broker)
        b.mt5 = SimpleNamespace(terminal_info=lambda t=terminal: t, account_info=lambda c=compte: c)
        assert b.can_trade() is attendu
