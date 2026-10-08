"""
════════════════════════════════════════════════════════════════════
JOURNAL DE TRADING STRUCTURÉ (SQLite)
deriv/trading_journal.py
════════════════════════════════════════════════════════════════════

Chaque trade est enregistré avec son contexte complet (régime HMM,
votes des 7 modèles, session, ADX/RSI/ATR, phase de la stratégie) pour :
  - analyser les patterns gain/perte (journal_learner.py)
  - identifier les conditions optimales (session, régime, seuils)
  - ré-entraîner les modèles sur les vrais résultats (export_for_ml)

DB configurable via JOURNAL_DB (défaut : trading_journal.db à la racine).
Deux modes de clôture :
  - contrats binaires Deriv : pnl_dollar (→ pnl_pct = pnl/stake)
  - trades SL/TP (backtest/stratégie) : exit_price (→ pnl_pct sur le prix)
"""
from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional
from deriv.constants import Config

import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = Config.JOURNAL_DB or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "trading_journal.db")
)


@dataclass
class JournalEntry:
    # Identifiant
    trade_id:        str
    # Setup du trade
    timestamp_entry: str           # ISO 8601 (UTC)
    symbol:          str
    direction:       str           # "long" / "short"
    entry_price:     float
    stop_loss:       float         # 0/±0.1% si option binaire (SL/TP N/A)
    take_profit:     float
    risk_reward:     float
    stake:           float         # montant misé (options binaires)
    # Contexte marché à l'entrée
    regime:          str           # HMM : Bull / Bear / Range
    hmm_confidence:  float         # 0-1
    adx:             float
    rsi:             float
    atr:             float
    ema_fast:        float
    ema_slow:        float
    volume_ratio:    float
    session:         str           # london / newyork / asian / overlap
    # Votes des 7 modèles de l'ensemble
    vote_hmm:        float
    vote_xgboost:    float
    vote_lstm:       float
    vote_kalman:     float
    vote_rsi:        float
    vote_trend:      float
    vote_momentum:   float
    ensemble_votes:  int           # votes concordants avec le signal
    ensemble_signal: str           # BUY / SELL / HOLD
    # Phase de la stratégie edge (ENSEMBLE = trade de l'ensemble seul)
    strategy_phase:  str
    pullback_candles: int
    rationale:       str
    # Coût du spread (Boom/Crash ≈ 0.2% aller-retour)
    spread_cost_pct: float = 0.0
    # PnL net après déduction du spread
    pnl_net_pct: Optional[float] = None
    exit_price:      Optional[float] = None
    pnl_dollar:      Optional[float] = None
    pnl_pct:         Optional[float] = None
    duration_bars:   Optional[int] = None
    exit_reason:     Optional[str] = None  # tp/sl/contract_win/contract_loss
    # Évaluation post-trade
    quality_score:   Optional[int] = None   # 1-5
    notes:           Optional[str] = None
    lessons:         Optional[str] = None


_COLS = list(JournalEntry.__dataclass_fields__)


class TradingJournal:
    """Journal persisté en SQLite, exploitable par le modèle ML."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DB_PATH
        self._init_db()

    def _init_db(self):
        """Crée les tables/index si absents (idempotent)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id         TEXT PRIMARY KEY,
                    timestamp_entry  TEXT NOT NULL,
                    symbol           TEXT NOT NULL,
                    direction        TEXT NOT NULL,
                    entry_price      REAL NOT NULL,
                    stop_loss        REAL,
                    take_profit      REAL,
                    risk_reward      REAL,
                    stake            REAL,
                    regime           TEXT,
                    hmm_confidence   REAL,
                    adx              REAL,
                    rsi              REAL,
                    atr              REAL,
                    ema_fast         REAL,
                    ema_slow         REAL,
                    volume_ratio     REAL,
                    session          TEXT,
                    vote_hmm         REAL,
                    vote_xgboost     REAL,
                    vote_lstm        REAL,
                    vote_kalman      REAL,
                    vote_rsi         REAL,
                    vote_trend       REAL,
                    vote_momentum    REAL,
                    ensemble_votes   INTEGER,
                    ensemble_signal  TEXT,
                    strategy_phase   TEXT,
                    pullback_candles INTEGER,
                    rationale        TEXT,
                    timestamp_exit   TEXT,
                    exit_price       REAL,
                    pnl_dollar       REAL,
                    pnl_pct          REAL,
                    duration_bars    INTEGER,
                    exit_reason      TEXT,
                    spread_cost_pct  REAL,
                    pnl_net_pct      REAL,
                    quality_score    INTEGER,
                    notes            TEXT,
                    lessons          TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_regime ON trades(regime)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date         TEXT PRIMARY KEY,
                    symbol       TEXT,
                    n_trades     INTEGER,
                    n_wins       INTEGER,
                    pnl_total    REAL,
                    win_rate     REAL,
                    avg_rr       REAL,
                    regime_modal TEXT
                )
            """)
            conn.commit()
            # Migration idempotente : ajoute les colonnes spread/pnl_net
            # si la table existait AVANT (CREATE IF NOT EXISTS ne migre pas).
            cols = {r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()}
            if "spread_cost_pct" not in cols:
                conn.execute(
                    "ALTER TABLE trades ADD COLUMN spread_cost_pct REAL DEFAULT 0.0")
            if "pnl_net_pct" not in cols:
                conn.execute(
                    "ALTER TABLE trades ADD COLUMN pnl_net_pct REAL")
            conn.commit()

    # ── Écriture : ouverture ───────────────────────────────────────
    def log_entry(self, entry: JournalEntry) -> str:
        """Enregistre un trade à l'ouverture (INSERT OR REPLACE)."""
        d = asdict(entry)
        cols = ", ".join(d.keys())
        placeholders = ", ".join("?" * len(d))
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO trades ({cols}) VALUES ({placeholders})",
                list(d.values()),
            )
            conn.commit()
        logger.info(f"Trade journalisé : {entry.trade_id}")
        return entry.trade_id

    @staticmethod
    def _fetch_open(conn: sqlite3.Connection, trade_id: str) -> tuple:
        """Récupère une position ouverte ; lève ValueError si introuvable."""
        row = conn.execute(
            "SELECT entry_price, direction, stake, timestamp_entry "
            "FROM trades WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Trade {trade_id} non trouvé dans le journal")
        if row[0] is None:
            raise ValueError(f"Trade {trade_id} : entry_price manquant")
        return row

    @staticmethod
    def _duration_h4_bars(ts_entry: str | None) -> int | None:
        """Durée approximative en barres H4 depuis l'entrée (None si incalculable)."""
        try:
            dt = datetime.fromisoformat(str(ts_entry))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (datetime.now(timezone.utc) - dt).total_seconds()
            return max(0, int(delta // 14400))
        except Exception:
            return None

    @staticmethod
    def _apply_exit(conn: sqlite3.Connection, trade_id: str,
                    exit_price: float | None, pnl_dollar: float,
                    pnl_pct: float, exit_reason: str,
                    duration_bars: int | None, notes: str) -> None:
        # Récupère le coût du spread enregistré à l'ouverture
        row = conn.execute(
            "SELECT spread_cost_pct FROM trades WHERE trade_id = ?",
            (trade_id,),
        ).fetchone()
        spread = float(row[0]) if row and row[0] is not None else 0.0
        pnl_net = pnl_pct - spread
        conn.execute(
            """
            UPDATE trades SET
                timestamp_exit = ?,
                exit_price     = ?,
                pnl_dollar     = ?,
                pnl_pct        = ?,
                pnl_net_pct    = ?,
                exit_reason    = ?,
                duration_bars  = ?,
                notes          = COALESCE(notes || ' | ', '') || ?
            WHERE trade_id = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                exit_price, round(float(pnl_dollar), 4),
                round(float(pnl_pct), 3), round(pnl_net, 3),
                exit_reason,
                duration_bars, notes or "", trade_id,
            ),
        )
        conn.commit()

    # ── Clôture SL/TP (prix → pnl en %) ────────────────────────────
    def log_exit(self, trade_id: str, exit_price: float,
                 exit_reason: str, notes: str = "") -> dict:
        """Clôture prix (backtest/stratégie) : pnl_pct calculé sur le prix."""
        with sqlite3.connect(self.db_path) as conn:
            entry_price, direction, stake, ts_entry = self._fetch_open(
                conn, trade_id)
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_dollar = (stake or 0.0) * pnl_pct / 100.0
            row = conn.execute(
                "SELECT spread_cost_pct FROM trades WHERE trade_id = ?",
                (trade_id,)).fetchone()
            spread = float(row[0]) if row and row[0] is not None else 0.0
            self._apply_exit(conn, trade_id, exit_price, pnl_dollar,
                             pnl_pct, exit_reason,
                             self._duration_h4_bars(ts_entry), notes)
        logger.info(f"Trade clôturé : {trade_id} | PnL {pnl_pct:+.2f}% ({exit_reason})")
        return {"trade_id": trade_id, "pnl_pct": round(pnl_pct, 3),
                "pnl_net_pct": round(pnl_pct - spread, 3),
                "pnl_dollar": round(pnl_dollar, 4), "exit_reason": exit_reason}

    # ── Clôture contrat binaire Deriv (pnl $ → pnl en % du stake) ───
    def log_exit_contract(self, trade_id: str, pnl_dollar: float,
                          notes: str = "") -> dict:
        """Clôture contrat binaire : pnl_pct = pnl_dollar / stake × 100."""
        with sqlite3.connect(self.db_path) as conn:
            _, _, stake, ts_entry = self._fetch_open(conn, trade_id)
            stake = stake or 0.0
            pnl_pct = (pnl_dollar / stake * 100.0) if stake > 0 else 0.0
            exit_reason = "contract_win" if pnl_dollar > 0 else "contract_loss"
            row = conn.execute(
                "SELECT spread_cost_pct FROM trades WHERE trade_id = ?",
                (trade_id,)).fetchone()
            spread = float(row[0]) if row and row[0] is not None else 0.0
            self._apply_exit(conn, trade_id, None, pnl_dollar,
                             pnl_pct, exit_reason,
                             self._duration_h4_bars(ts_entry), notes)
        logger.info(f"Contrat clôturé : {trade_id} | PnL {pnl_dollar:+.2f}$ ({exit_reason})")
        return {"trade_id": trade_id, "pnl_pct": round(pnl_pct, 3),
                "pnl_net_pct": round(pnl_pct - spread, 3),
                "pnl_dollar": round(float(pnl_dollar), 4),
                "exit_reason": exit_reason}

    # ── Lecture : statistiques ─────────────────────────────────────
    def get_stats(self, symbol: str | None = None,
                  regime: str | None = None,
                  last_n: int | None = None) -> dict:
        """Métriques sur les trades clôturés (filtres optionnels)."""
        query = (
            "SELECT trade_id, direction, pnl_dollar, pnl_pct, regime,"
            " hmm_confidence, ensemble_votes, exit_reason, risk_reward,"
            " session, adx, rsi, pullback_candles FROM trades"
            " WHERE timestamp_exit IS NOT NULL AND pnl_pct IS NOT NULL"
        )
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if regime:
            query += " AND regime = ?"
            params.append(regime)
        query += " ORDER BY timestamp_entry DESC"
        if last_n:
            query += f" LIMIT {int(last_n)}"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        if not rows:
            return {"message": "Aucun trade clôturé", "n_trades": 0}

        df = pd.DataFrame(rows, columns=[
            "trade_id", "direction", "pnl_dollar", "pnl_pct", "regime",
            "hmm_confidence", "ensemble_votes", "exit_reason", "risk_reward",
            "session", "adx", "rsi", "pullback_candles",
        ])
        wins = df[df["pnl_pct"] > 0]
        losses = df[df["pnl_pct"] < 0]
        gross_profit = float(wins["pnl_pct"].sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses["pnl_pct"].sum())) if len(losses) else 0.0

        par_regime: dict = {}
        for r in df["regime"].dropna().unique():
            sub = df[df["regime"] == r]
            par_regime[str(r)] = {
                "win_rate": float((sub["pnl_pct"] > 0).mean()),
                "n_trades": int(len(sub)),
                "pnl_moyen": float(sub["pnl_pct"].mean()),
            }
        par_session: dict = {}
        for s in df["session"].dropna().unique():
            sub = df[df["session"] == s]
            par_session[str(s)] = {
                "win_rate": float((sub["pnl_pct"] > 0).mean()),
                "n_trades": int(len(sub)),
            }
        par_votes: dict = {}
        for v in sorted(df["ensemble_votes"].dropna().unique()):
            sub = df[df["ensemble_votes"] == v]
            par_votes[int(v)] = {
                "win_rate": float((sub["pnl_pct"] > 0).mean()),
                "n_trades": int(len(sub)),
            }
        return {
            "n_trades": int(len(df)),
            "n_wins": int(len(wins)),
            "n_losses": int(len(losses)),
            "win_rate": round(float(len(wins) / len(df)), 3),
            "pnl_total_dollar": round(float(df["pnl_dollar"].sum()), 2),
            "pnl_total_pct": round(float(df["pnl_pct"].sum()), 2),
            "pnl_moyen_pct": round(float(df["pnl_pct"].mean()), 3),
            "profit_factor": round(gross_profit / max(gross_loss, 1e-9), 2),
            "rr_moyen": round(float(df["risk_reward"].mean()), 2),
            "par_votes": par_votes,
            "par_regime": par_regime,
            "par_session": par_session,
            "breakeven_requis": "55.6% (options binaires Deriv)",
        }

    # ── Export ML : trades clôturés → features + label ─────────────
    def export_for_ml(self) -> pd.DataFrame:
        """DataFrame d'entraînement : won = 1 si trade gagnant."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql(
                "SELECT * FROM trades WHERE timestamp_exit IS NOT NULL"
                " AND pnl_pct IS NOT NULL",
                conn,
            )
        if df.empty:
            return df
        df["won"] = (df["pnl_pct"] > 0).astype(int)
        df["regime_encoded"] = df["regime"].map(
            {"Bull": 1, "Range": 0, "Bear": -1})
        df["session_encoded"] = df["session"].map(
            {"asian": 0, "newyork": 1, "london": 2, "overlap": 3})
        feature_cols = [
            "hmm_confidence", "adx", "rsi", "atr", "volume_ratio",
            "vote_hmm", "vote_xgboost", "vote_lstm", "vote_kalman",
            "vote_rsi", "vote_trend", "vote_momentum",
            "ensemble_votes", "risk_reward", "pullback_candles",
        ]
        keep = [c for c in feature_cols if c in df.columns]
        return df[keep + ["regime_encoded", "session_encoded",
                          "won", "pnl_pct", "pnl_dollar"]]
