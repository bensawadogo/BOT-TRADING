"""Gestion du capital et des règles de risque pour Deriv."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from deriv.constants import Config

from sqlite3 import OperationalError

from deriv.constants import BREAKEVEN_WINRATE as _BREAKEVEN_WINRATE


@dataclass
class StakeDecision:
    allowed: bool
    stake: float
    mode: str  # auto | manual | blocked
    message: str


@dataclass
class TradeResult:
    """Résultat d'un contrat exécuté par le bot (ensemble 4/5)."""
    symbol: str
    pnl: float
    contract_type: Optional[str] = None
    stake: Optional[float] = None
    timestamp: Optional[float] = None


logger = logging.getLogger(__name__)


class RiskManager:
    """
    Règles de gestion du capital pour Deriv.
    Basé sur les bonnes pratiques des traders professionnels.
    """

    MAX_STAKE_PCT = 0.02  # Max 2% du capital par trade
    MAX_DAILY_LOSS = 0.10  # Stop si -10% en 1 jour
    MAX_CONSECUTIVE = 3  # Stop après 3 pertes consécutives
    COOLDOWN_MINUTES = 15  # Pause après stop
    BREAKEVEN_WINRATE = _BREAKEVEN_WINRATE  # importé depuis deriv.constants (source unique)

    def __init__(
        self,
        capital: float = 1000.0,
        auto_max: Optional[float] = None,
        manual_threshold: Optional[float] = None,
    ):
        self.capital = float(capital)
        self.daily_loss = 0.0
        self.consecutive_loss = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.auto_max = float(auto_max or Config.AUTO_MAX_STAKE)
        self.manual_threshold = float(
            manual_threshold or Config.MANUAL_THRESHOLD
        )
        self.account_type = Config.ACCOUNT_TYPE.lower()
        self._net_pnl = 0.0

        # Paramètres surchargables via .env (mission ensemble 4/5)
        self.max_risk_pct = Config.MAX_RISK_PCT
        self.daily_stop_pct = float(
            Config.DAILY_STOP_PCT
        )
        self.max_consecutive = int(
            Config.MAX_CONSECUTIVE
        )

        # État temporel
        self.cooldown_until: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self._day: date = date.today()   # jour auquel se rapporte daily_loss

    def _roll_day(self) -> None:
        """Nouveau jour → la perte journalière repart de zéro."""
        today = date.today()
        if today != self._day:
            self._day = today
            self.daily_loss = 0.0

    def max_stake(self) -> float:
        """Calcule le stake maximum autorisé (2% du capital par défaut)."""
        return round(self.capital * self.max_risk_pct, 2)

    # ── Kelly Criterion fractionnaire (V2) ───────────────────────────
    KELLY_FRACTION = 0.25  # Kelly partiel 25% — recommandation académique

    def kelly_stake(self, win_rate: float, payout: float,
                    fraction: float | None = None) -> float:
        """
        Mise optimale via le Critère de Kelly, pour options binaires.

        Pour un pari binaire (gagne +payout×stake / perd −stake) :
            K = p − (1 − p) / payout
        avec p = win_rate réel, payout = gain net relatif (ex 0.85 = +85%).

        fraction=0.25 → Kelly partiel (quart de Kelly) : réduit la variance
        sans sacrifier beaucoup d'espérance — standard académique.

        Retourne la FRACTION du capital à risquer (0.0 si Kelly ≤ 0).
        """
        fraction = self.KELLY_FRACTION if fraction is None else fraction
        p = float(win_rate)
        payout = float(payout)
        if payout <= 0 or p <= 0 or p >= 1:
            return 0.0
        kelly_full = p - (1.0 - p) / payout
        if kelly_full <= 0:
            return 0.0   # espérance négative → ne pas trader
        kelly_partial = kelly_full * fraction
        # Toujours borné par le risque max par trade
        return float(min(kelly_partial, self.max_risk_pct))

    def get_optimal_stake(self, win_rate_running: float | None = None,
                          payout: float = 0.85) -> tuple[float, str]:
        """
        Stake optimal : Kelly 25% sur le WR réel du journal, borné par
        [0.35$ (min Deriv), min(2% capital, auto_max)].

        Retourne (stake_usd, raison). Repli : stake auto si < 20 trades
        dans le journal (échantillon insuffisant pour estimer le WR).
        """
        wr = win_rate_running
        if wr is None:
            try:
                from deriv.trading_journal import TradingJournal
                stats = TradingJournal().get_stats()
                if int(stats.get("n_trades", 0)) < 20:
                    wr = None
                else:
                    wr = float(stats["win_rate"])
            except (ImportError, OperationalError) as exc:
                logger.warning(f"Journal inaccessible : {exc}")
                wr = None
        if wr is None or wr <= 0:
            return (float(min(self.max_stake(), self.auto_max)),
                    "échantillon insuffisant → stake auto par défaut")
        frac = self.kelly_stake(wr, payout)
        if frac <= 0:
            return (0.0, f"Kelly négatif (WR {wr:.1%} / payout {payout}) "
                         "→ ne pas trader")
        stake_usd = self.capital_actuel * frac
        limit = float(min(self.max_stake(), self.auto_max))
        return (max(0.35, round(min(stake_usd, limit), 2)),
                f"Kelly 25% (WR {wr:.1%}, payout {payout}) → {frac:.2%} capital")

    def can_trade(self) -> tuple[bool, str]:
        """Vérifie si on peut trader selon les règles de risque."""
        self._roll_day()
        if self.daily_loss >= self.capital * self.daily_stop_pct:
            # Stop journalier : reprise le lendemain (_roll_day).
            return False, f"Perte journalière max atteinte ({self.daily_loss:.2f}$)"
        if self.consecutive_loss >= self.max_consecutive:
            if self.cooldown_until is None:
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.COOLDOWN_MINUTES)
            if datetime.now() < self.cooldown_until:
                return (
                    False,
                    f"{self.max_consecutive} pertes consécutives — pause {self.COOLDOWN_MINUTES}min",
                )
            # Pause terminée : on repart (avant : blocage définitif, le
            # compteur n'étant remis à zéro que par un trade gagnant).
            self.consecutive_loss = 0
            self.cooldown_until = None
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            rest = int((self.cooldown_until - datetime.now()).total_seconds() // 60)
            return False, f"Refroidissement : reprise dans {max(rest, 1)} min"
        return True, "OK"

    def record_result(self, profit_loss: float):
        """Enregistre le résultat d'un trade."""
        self._roll_day()
        self.total_trades += 1
        self.last_trade_time = datetime.now()
        if profit_loss < 0:
            self.daily_loss += abs(profit_loss)
            self.consecutive_loss += 1
            # Déclenche un cooldown dès qu'un stop est atteint
            if (self.consecutive_loss >= self.max_consecutive
                    or self.daily_loss >= self.capital * self.daily_stop_pct):
                self.cooldown_until = datetime.now() + timedelta(
                    minutes=self.COOLDOWN_MINUTES
                )
        else:
            self.winning_trades += 1
            self.consecutive_loss = 0
        self._net_pnl += profit_loss

    @property
    def winrate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    # Fusion : check() centralise les règles de risque utilisées par le bot ;
    # decide() conserve l'interface StakeDecision comme alias de compatibilité.
    def check(
        self,
        requested_stake: float,
        confiance_pct: float = 0.0,
        confirmed: bool = False,
    ) -> tuple[bool, str]:
        """
        Vérifie si un trade est autorisé (stake + stop + cooldown).
        Retourne (ok, raison). Raison = 'CONFIRMATION_REQUISE…' quand le
        stake dépasse le max auto (le bot doit alors lever une alerte).
        """
        can, reason = self.can_trade()
        if not can:
            return False, reason

        confiance = float(confiance_pct or 0.0)
        if confiance and confiance < self.BREAKEVEN_WINRATE:
            return (
                False,
                f"Confiance {confiance:.1f}% < breakeven "
                f"{self.BREAKEVEN_WINRATE:.1f}% — trade refusé",
            )

        stake = max(0.1, float(requested_stake))
        limit = min(self.max_stake(), self.auto_max)
        if stake > limit and not confirmed:
            return (
                False,
                f"CONFIRMATION_REQUISE: stake {stake:.2f}$ > max auto {limit:.2f}$",
            )
        return True, "OK"

    def decide(self, requested_stake: float, confirmed: bool = False) -> StakeDecision:
        """Compatibilité : decide() délègue à check() puis formate la réponse."""
        stake = max(0.35, float(requested_stake))
        ok, reason = self.check(stake, confirmed=confirmed)
        if not ok:
            mode = "manual" if "CONFIRMATION_REQUISE" in reason else "blocked"
            return StakeDecision(allowed=False, stake=stake, mode=mode, message=reason)

        limit = min(self.max_stake(), self.auto_max)
        if self.account_type != "demo" and not confirmed and stake > self.manual_threshold:
            return StakeDecision(
                allowed=False,
                stake=stake,
                mode="manual",
                message="Compte réel : confirmation manuelle requise au-dessus du seuil.",
            )
        if stake > limit:
            return StakeDecision(
                allowed=True,
                stake=stake,
                mode="manual",
                message="Stake manuel confirmé.",
            )
        return StakeDecision(
            allowed=True,
            stake=stake,
            mode="auto",
            message="Stake auto autorisé.",
        )

    def record(self, result) -> None:
        """
        Enregistre un résultat de trade (TradeResult ou float pnl).
        Met à jour la perf et le cooldown si un stop est atteint.
        """
        if isinstance(result, (int, float)):
            self.record_result(float(result))
            return
        pnl = getattr(result, "pnl", 0.0)
        self.record_result(float(pnl))

    @property
    def capital_actuel(self) -> float:
        """Capital après les pertes et les gains du jour."""
        return round(self.capital + self._net_pnl, 2)

    @property
    def stats(self) -> dict:
        """Statistiques de session pour messages Telegram / tableau de bord."""
        return {
            "winrate_jour": round(self.winrate * 100, 1),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "daily_loss": round(self.daily_loss, 2),
            "consecutive_loss": self.consecutive_loss,
            "capital_actuel": self.capital_actuel,
            "in_cooldown": bool(
                self.cooldown_until and datetime.now() < self.cooldown_until
            ),
        }
