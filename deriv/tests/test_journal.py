"""
Tests du journal de trading SQLite (mission edge + journal).
Lance avec : python -m pytest deriv/tests/test_journal.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

from deriv.trading_journal import JournalEntry, TradingJournal


def make_entry(trade_id: str = "T001") -> JournalEntry:
    """Entrée complète conforme au schéma (7 modèles de l'ensemble)."""
    return JournalEntry(
        trade_id=trade_id,
        timestamp_entry="2026-09-12T04:00:00+00:00",
        symbol="frxEURUSD",
        direction="long",
        entry_price=1.1000,
        stop_loss=0.0,           # option binaire : SL/TP N/A
        take_profit=0.0,
        risk_reward=0.0,
        stake=2.0,
        regime="Bull",
        hmm_confidence=0.82,
        adx=28.5,
        rsi=45.0,
        atr=0.0012,
        ema_fast=1.0998,
        ema_slow=1.0990,
        volume_ratio=1.2,
        session="london",
        vote_hmm=1,
        vote_xgboost=1,
        vote_lstm=0.5,
        vote_kalman=1,
        vote_rsi=1,
        vote_trend=1,
        vote_momentum=0.5,
        ensemble_votes=5,
        ensemble_signal="BUY",
        strategy_phase="ENSEMBLE",
        pullback_candles=0,
        rationale="Ensemble BUY 5/7",
    )


@pytest.fixture
def journal(tmp_path):
    return TradingJournal(db_path=str(tmp_path / "test_journal.db"))


class TestJournal:

    def test_log_entry_pas_encore_cloture(self, journal):
        journal.log_entry(make_entry())
        stats = journal.get_stats()
        # Pas encore de trade clôturé → stats vides
        assert stats["n_trades"] == 0

    def test_log_exit_contract_win(self, journal):
        journal.log_entry(make_entry())
        res = journal.log_exit_contract("T001", pnl_dollar=1.8)
        assert res["pnl_pct"] > 0          # 1.8$ / 2$ stake = +90%
        assert res["exit_reason"] == "contract_win"

    def test_log_exit_contract_loss(self, journal):
        journal.log_entry(make_entry())
        res = journal.log_exit_contract("T001", pnl_dollar=-2.0)
        assert res["pnl_pct"] < 0          # perte totale du stake = -100%
        assert res["exit_reason"] == "contract_loss"

    def test_log_exit_sl_tp(self, journal):
        journal.log_entry(make_entry())
        res = journal.log_exit("T001", exit_price=1.1050, exit_reason="tp")
        assert res["pnl_pct"] > 0          # long 1.1000 → 1.1050
        stats = journal.get_stats()
        assert stats["n_trades"] == 1

    def test_trade_inconnu_leve_erreur(self, journal):
        with pytest.raises(ValueError):
            journal.log_exit_contract("INCONNU", pnl_dollar=1.0)
        with pytest.raises(ValueError):
            journal.log_exit("INCONNU", exit_price=1.1, exit_reason="tp")

    def test_stats_win_rate_et_profit_factor(self, journal):
        # 2 wins (+90% stake), 1 loss (-100%) → WR 2/3, PF (180/100)
        for i, pnl in enumerate((1.8, 1.8, -2.0)):
            journal.log_entry(make_entry(f"T{i:03d}"))
            journal.log_exit_contract(f"T{i:03d}", pnl_dollar=pnl)
        stats = journal.get_stats()
        assert stats["n_trades"] == 3
        assert stats["n_wins"] == 2
        assert stats["win_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert stats["profit_factor"] == pytest.approx(1.8, abs=0.01)
        assert stats["pnl_total_dollar"] == pytest.approx(1.6, abs=0.01)

    def test_stats_par_regime_et_session(self, journal):
        for i, regime in enumerate(("Bull", "Bull", "Bear")):
            e = make_entry(f"R{i:03d}")
            e.regime = regime
            e.session = "london" if regime == "Bull" else "asian"
            journal.log_entry(e)
            journal.log_exit_contract(f"R{i:03d}", pnl_dollar=1.8)
        stats = journal.get_stats()
        assert stats["par_regime"]["Bull"]["n_trades"] == 2
        assert stats["par_regime"]["Bull"]["win_rate"] == 1.0
        assert stats["par_session"]["asian"]["n_trades"] == 1

    def test_export_ml_structure(self, journal):
        journal.log_entry(make_entry())
        journal.log_exit_contract("T001", pnl_dollar=1.8)
        df = journal.export_for_ml()
        assert len(df) == 1
        assert "won" in df.columns
        assert int(df["won"].iloc[0]) == 1
        for col in ("hmm_confidence", "ensemble_votes", "regime_encoded",
                    "session_encoded", "pnl_pct"):
            assert col in df.columns, f"Colonne ML manquante : {col}"

    def test_export_ml_vide(self, journal):
        df = journal.export_for_ml()
        assert df.empty

    def test_persistance_reouverture(self, journal, tmp_path):
        journal.log_entry(make_entry())
        journal.log_exit_contract("T001", pnl_dollar=-2.0)
        # Ré-ouvre la même DB → le trade clôturé est toujours là
        journal2 = TradingJournal(
            db_path=str(tmp_path / "test_journal.db"))
        stats = journal2.get_stats()
        assert stats["n_trades"] == 1
        assert stats["n_losses"] == 1