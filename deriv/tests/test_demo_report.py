"""Le verdict de démo applique le protocole : 200 trades, p < 0.05, Wilson > seuil."""
import pytest

from deriv import demo_report as dr
from deriv.trading_journal import JournalEntry, TradingJournal


def _trades(wins, losses, stake=2.0, payout=0.95):
    return [(stake * payout, stake, "")] * wins + [(-stake, stake, "")] * losses


def test_seuil_calcule_sur_les_vrais_payouts():
    r = dr.verdict(_trades(60, 40, payout=0.95))
    assert r["payout_moyen"] == pytest.approx(0.95)
    assert r["seuil_rentabilite"] == pytest.approx(1 / 1.95)


def test_en_cours_tant_que_moins_de_200_trades():
    assert dr.verdict(_trades(90, 10))["decision"].startswith("EN COURS")


def test_edge_confirme():
    r = dr.verdict(_trades(150, 70, payout=0.95))      # 68 % >> 51.3 %
    assert r["decision"].startswith("EDGE CONFIRMÉ")


def test_pas_d_edge():
    r = dr.verdict(_trades(80, 140, payout=0.95))      # 36 %
    assert r["decision"].startswith("PAS D'EDGE")


def test_non_concluant():
    r = dr.verdict(_trades(115, 100, payout=0.95))     # 53.5 % ≈ seuil
    assert r["decision"].startswith("NON CONCLUANT")


def test_lit_le_journal_reel(tmp_path):
    db = str(tmp_path / "j.db")
    journal = TradingJournal(db_path=db)
    for i, pnl in enumerate([1.9, -2.0, 1.9]):
        champs = {f: 0.0 for f in JournalEntry.__dataclass_fields__}
        champs.update(trade_id=f"T{i}", timestamp_entry="2026-01-01T00:00:00",
                      symbol="BOOM500", direction="short", stake=2.0,
                      regime="spike_drift", session="asian", ensemble_votes=0,
                      ensemble_signal="DRIFT", strategy_phase=dr.PHASE,
                      pullback_candles=0, rationale="t",
                      pnl_net_pct=None, exit_price=None, pnl_dollar=None,
                      pnl_pct=None, duration_bars=None, exit_reason=None)
        champs = {k: v for k, v in champs.items() if k in JournalEntry.__init__.__code__.co_varnames}
        journal.log_entry(JournalEntry(**champs))
        journal.log_exit_contract(f"T{i}", pnl_dollar=pnl)
    r = dr.main(["--db", db, "--symbol", "BOOM500"])
    assert r["n_trades"] == 3 and r["n_wins"] == 2
