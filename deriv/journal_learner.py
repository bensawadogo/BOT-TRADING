"""
════════════════════════════════════════════════════════════════════
BOUCLE D'AMÉLIORATION — le modèle apprend du journal
deriv/journal_learner.py
════════════════════════════════════════════════════════════════════

Chaque semaine (ou après 10+ trades), analyse le journal pour :
  1. identifier les conditions qui gagnent (régime, session, votes)
  2. proposer les seuils optimaux (confiance HMM, ADX, votes min)
  3. générer un rapport + message Telegram de recommandations

⚠️  Les recommandations sont INFO : la règle 4/5 est VERROUILLÉE
    (MIN_VOTES_TO_TRADE, tests test_regle_4_sur_5). Tout changement
    se fait manuellement et doit passer la suite de tests.
"""
from __future__ import annotations

import logging

import pandas as pd

from deriv.trading_journal import TradingJournal

logger = logging.getLogger(__name__)


class JournalLearner:
    """Analyse le journal et génère des recommandations concrètes."""

    MIN_TRADES_ANALYSE = 10

    def __init__(self, journal: TradingJournal | None = None):
        self.journal = journal or TradingJournal()

    def analyse_et_recommande(self) -> dict:
        """Analyse complète du journal → recommandations."""
        df = self.journal.export_for_ml()
        if len(df) < self.MIN_TRADES_ANALYSE:
            return {
                "status": "Pas assez de trades",
                "n_requis": self.MIN_TRADES_ANALYSE,
                "n_actuels": int(len(df)),
            }

        recommandations: list[str] = []
        seuils: dict = {}

        # ── 1. Seuil optimal de votes ensemble ───────────────────────
        wr_par_votes: dict = {}
        for v in (2, 3, 4, 5):
            sub = df[df["ensemble_votes"] >= v]
            if len(sub) >= 5:
                wr_par_votes[v] = {
                    "wr": float(sub["won"].mean()),
                    "n": int(len(sub)),
                }
        if wr_par_votes:
            best = max(
                wr_par_votes.items(),
                key=lambda x: (x[1]["wr"], x[1]["n"]),
            )
            seuils["min_votes"] = best[0]
            if best[0] != 4:
                recommandations.append(
                    f"Votes min : 4/5 verrouillé — observation : {best[0]}+ "
                    f"votes → WR {best[1]['wr']:.1%} sur {best[1]['n']} "
                    "trades (changement manuel + tests requis)"
                )

        # ── 2. Confiance HMM optimale ────────────────────────────────
        best_conf, best_conf_wr = 0.70, -1.0
        for t in (0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            sub = df[df["hmm_confidence"] >= t]
            if len(sub) >= 5:
                wr = float(sub["won"].mean())
                if wr > best_conf_wr:
                    best_conf_wr, best_conf = wr, t
        seuils["hmm_confidence_min"] = best_conf
        if best_conf_wr >= 0 and abs(best_conf - 0.70) > 0.05:
            recommandations.append(
                f"Confiance HMM min : passer de 0.70 à {best_conf:.2f} "
                f"(WR {best_conf_wr:.1%} sur les trades filtrés)"
            )

        # ── 3. Meilleure session de trading ──────────────────────────
        if "session_encoded" in df.columns:
            sessions = {0: "asian", 1: "newyork", 2: "london", 3: "overlap"}
            stats_session: dict = {}
            for enc, nom in sessions.items():
                sub = df[df["session_encoded"] == enc]
                if len(sub) >= 3:
                    stats_session[nom] = {
                        "win_rate": float(sub["won"].mean()),
                        "n": int(len(sub)),
                    }
            if stats_session:
                best_s = max(stats_session.items(),
                             key=lambda x: x[1]["win_rate"])
                seuils["meilleure_session"] = best_s[0]
                recommandations.append(
                    f"Meilleure session : {best_s[0]} "
                    f"({best_s[1]['win_rate']:.1%} WR sur "
                    f"{best_s[1]['n']} trades)"
                )

        # ── 4. ADX minimum optimal ───────────────────────────────────
        best_adx, best_adx_wr = 20, -1.0
        for t in (15, 20, 25, 30):
            sub = df[df["adx"] >= t]
            if len(sub) >= 5:
                wr = float(sub["won"].mean())
                if wr > best_adx_wr:
                    best_adx_wr, best_adx = wr, t
        seuils["adx_min"] = best_adx
        if best_adx_wr >= 0 and best_adx != 20:
            recommandations.append(
                f"ADX min optimal observé : {best_adx} "
                f"(WR {best_adx_wr:.1%} sur les trades filtrés)"
            )

        # ── 5. Régimes effectifs ─────────────────────────────────────
        stats_regime: dict = {}
        if "regime_encoded" in df.columns:
            enc_regimes = {1: "Bull", 0: "Range", -1: "Bear"}
            for enc, nom in enc_regimes.items():
                sub = df[df["regime_encoded"] == enc]
                if len(sub) > 0:
                    stats_regime[nom] = {
                        "win_rate": float(sub["won"].mean()),
                        "n": int(len(sub)),
                    }
            regimes_perdants = [
                n for n, s in stats_regime.items()
                if s["win_rate"] < 0.5 and s["n"] >= 5
            ]
            if regimes_perdants:
                recommandations.append(
                    "Régimes perdants (n>=5) : " + ", ".join(regimes_perdants)
                )

        # ── Rapport final ────────────────────────────────────────────
        wr_global = float(df["won"].mean())
        return {
            "date_analyse": pd.Timestamp.now(tz="UTC").isoformat(),
            "n_trades_analyses": int(len(df)),
            "win_rate_global": round(wr_global, 3),
            "pnl_moyen_pct": round(float(df["pnl_pct"].mean()), 3),
            "seuils_optimaux": seuils,
            "recommandations": recommandations,
            "stats_par_regime": stats_regime,
            "profitable": wr_global >= 0.556,
            "message_telegram": self._format_telegram(
                df, recommandations, seuils),
        }

    def _format_telegram(self, df: pd.DataFrame,
                         recommandations: list[str], seuils: dict) -> str:
        wr = float(df["won"].mean())
        n = len(df)
        emoji = "✅" if wr >= 0.556 else "⚠️"
        lines = [
            f"{emoji} *Rapport Journal Trading*",
            f"Trades analysés : {n} | Win Rate : {wr:.1%}",
            "Breakeven requis : 55.6% (options binaires)",
            "",
            "*Paramètres optimaux détectés :*",
        ]
        for k, v in seuils.items():
            lines.append(f"  • {k} : {v}")
        if recommandations:
            lines.append("")
            lines.append("*Recommandations :*")
            lines.extend(f"  → {r}" for r in recommandations[:3])
        return "\n".join(lines)
