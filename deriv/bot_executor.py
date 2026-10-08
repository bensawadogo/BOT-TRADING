"""
═══════════════════════════════════════════════════════════════════
ÉTAPE 3 — BOT D'EXÉCUTION DERIV
deriv/bot_executor.py
═══════════════════════════════════════════════════════════════════

Exécute les trades sur Deriv UNIQUEMENT si l'Ensemble (4/5) dit TRADE.
Intègre le Risk Manager. Alerte Telegram avant chaque exécution.

Utilise la couche existante deriv/client.py (deriv-sdk) et aiohttp
(pas httpx, non installé dans ce projet).
"""
from __future__ import annotations

import asyncio
import logging

from deriv.client import DerivConnection
from deriv.constants import Config
from deriv.data_collector import DerivDataCollector
from deriv.ensemble_predictor import EnsemblePredictor, session_filter
from deriv.journal_learner import JournalLearner
from deriv.online_learner import OnlineLearner
from deriv.risk_manager import RiskManager, TradeResult
from deriv.telegram import TelegramNotifier
from deriv.strategies.boom_crash_drift import BoomCrashDriftStrategy
from deriv.strategies.regime_momentum_strategy import (
    RegimeMomentumStrategy,
)
from deriv.trading_journal import JournalEntry, TradingJournal
from deriv.weekly_retrain import RetrainScheduler

logger = logging.getLogger("DERIV_BOT")


class DerivBotExecutor:
    """Bot Deriv : collecte → ensemble 4/5 → risk → exécution CALL/PUT."""

    SYMBOL = Config.SYMBOL
    DURATION = Config.DURATION              # 1 min
    STAKE_AUTO = Config.AUTO_STAKE  # bot-specific, < Config.AUTO_MAX_STAKE
    CYCLE_SEC = Config.CYCLE_SEC             # analyse / min

    def __init__(self, capital_usd: float | None = None):
        self.conn: DerivConnection | None = None
        self.ensemble = EnsemblePredictor()
        self.collector: DerivDataCollector | None = None
        self.risk = RiskManager(
            capital=float(capital_usd or Config.CAPITAL_USD)
        )
        self.telegram = TelegramNotifier(
            Config.TELEGRAM_BOT_TOKEN,
            Config.TELEGRAM_CHAT_ID,
        )
        self._stop = False
        # Ré-entraînement hebdomadaire automatique (mission robustesse)
        self.retrain = RetrainScheduler()
        # Stratégie edge (machine à états HMM) + journal + boucle d'apprentissage
        # (mission edge : jamais de crash si SQLite indisponible → try/except)
        try:
            self.strategy = RegimeMomentumStrategy()
            self.journal = TradingJournal()
            self.learner = JournalLearner(journal=self.journal)
            # V2 — Online learning : apprend après chaque trade fermé
            self.online = OnlineLearner()
        except Exception:
            logger.warning("Journal/stratégie indisponibles — mode dégradé",
                           exc_info=True)
            self.strategy = None
            self.journal = None
            self.learner = None
            self.online = None
        self.open_trades: dict = {}  # {trade_id: {"entry": JournalEntry}}
        # Stratégie Boom/Crash Drift (uniquement si symbole BOOM/CRASH)
        sym = self.SYMBOL.upper()
        self.boom_crash = (
            BoomCrashDriftStrategy(symbol=self.SYMBOL)
            if ("BOOM" in sym or "CRASH" in sym) else None
        )

    @property
    def _telegram_active(self) -> bool:
        return self.telegram.active

    # ── Garde-fous anti-leakage au démarrage ────────────────────────
    async def _run_startup_checks(self, df) -> bool:
        """
        GARDE-FOUS (López de Prado, 2018) — lance la détection de feature
        leakage avant d'autoriser le trading.
        Retourne True si tout est OK, False si un problème est détecté.
        """
        from deriv.ensemble_predictor import (
            FEATURE_COLS,
            build_features,
            build_labels,
        )
        from deriv.leakage_detector import detect_feature_leakage

        logger.info("=== Vérification anti-leakage au démarrage ===")

        # 0. Target alignée (architecture découplée : labels séparés)
        labels = build_labels(df)
        if labels is None or labels.isna().any():
            logger.error("Target contient des NaN — alignement incorrect")
            return False

        # 1. Check feature leakage (target injectée pour le détecteur)
        feat_df = build_features(df)
        if feat_df.empty:
            logger.error("Pas assez de données pour les garde-fous")
            return False
        feat_df = feat_df.assign(
            target=labels.reindex(feat_df.index)
        ).dropna(subset=["target"])
        if len(feat_df) < 80:
            logger.error(
                f"Alignement features/labels trop court ({len(feat_df)} lignes)"
            )
            return False

        leakage = detect_feature_leakage(feat_df, FEATURE_COLS)
        if leakage["n_leaked"] > 0:
            logger.error(f"FUITE DÉTECTÉE : {leakage['leaked_features']}")
            await self._telegram(
                "🚨 *GARDE-FOU ACTIVÉ* — Bot arrêté\n"
                f"Features suspectes : {leakage['leaked_features']}\n"
                "Vérifier build_features() avant de relancer."
            )
            return False

        logger.info(
            f"✅ Anti-leakage OK — {leakage['n_total']} features vérifiées"
        )

        # 2. Check target alignment : chaque label aligné doit être calculable
        common = feat_df.index.intersection(labels.index)
        if len(common) < 80:
            logger.error(
                f"Alignement features/labels trop court ({len(common)} lignes)"
            )
            return False

        logger.info("✅ Target alignment OK")
        return True

    # ── Démarrage ───────────────────────────────────────────────────
    async def start(self):
        """Connecte, entraîne si besoin, lance les deux boucles (collect/analyse)."""
        self.conn = DerivConnection(
            app_id=Config.APP_ID,
            api_token=Config.API_TOKEN,
        )

        # Sécurité absolue : demo d'abord, real uniquement si explicite.
        if not self.conn.is_demo and not Config.ALLOW_REAL:
            raise RuntimeError(
                "Compte REAL détecté : interdit. Mets DERIV_ACCOUNT_TYPE=demo "
                "ou autorise explicitement avec DERIV_ALLOW_REAL=1 (déconseillé)."
            )

        await self.conn.connect()
        logger.info(f"Connecté — mode : {'DEMO' if self.conn.is_demo else 'REAL'}")

        self.collector = DerivDataCollector(self.SYMBOL, self.DURATION,
                                            connection=self.conn)
        await self.collector.connect()

        # Historique pour init / entraînement des modèles
        df_history = await self.collector.get_candle_history(500)
        if df_history.empty:
            raise RuntimeError("Impossible de charger l'historique Deriv — "
                               "vérifie le symbole, le token et le réseau.")
        if not self.ensemble.is_trained():
            logger.info("Premiers entraînements des modèles (historique seulement)...")
            self.ensemble.train_all(df_history)

        # RÉ-ENTRAÎNEMENT HEBDOMADAIRE AUTOMATIQUE (mission robustesse) :
        # si le dernier entraînement date de plus de 7 jours, on ré-entraîne
        # tous les modèles sur les données fraîches. Non-bloquant :
        # une erreur ici ne doit jamais empêcher le bot de tourner.
        try:
            if self.retrain.needs_retrain():
                logger.info(
                    "Rée-entraînement hebdomadaire déclenché "
                    f"(dernier : {self.retrain.load_state().get('last_train', 'jamais')})"
                )
                rep = await self.retrain.retrain_async(df=df_history)
                if rep.get("ok"):
                    await self._telegram(
                        f"♻️ Ré-entraînement hebdomadaire OK\n"
                        f"{rep['n_bougies']} bougies {self.retrain.symbol}"
                    )
                else:
                    logger.warning(f"Ré-entraînement hebdo échoué : {rep}")
        except Exception as exc:
            logger.warning(f"Ré-entraînement hebdo ignoré : {exc}")

        # GARDE-FOUS ANTI-LEAKAGE (López de Prado, 2018) : le bot refuse de
        # trader si une feature voit le futur ou si la target est mal alignée.
        if not await self._run_startup_checks(df_history):
            raise RuntimeError(
                "Garde-fous anti-leakage en échec au démarrage — bot arrêté. "
                "Voir les logs DERIV_BOT ci-dessus."
            )

        # Sans token API, le streaming ticks live est refusé par le serveur :
        # on bascule en mode 'historique rafraîchi' (cf. _analyse_loop).
        raw_token = Config.API_TOKEN
        token_ok = bool(raw_token) and "REMPLACE_PAR_TON_TOKEN" not in raw_token
        if not token_ok:
            logger.warning(
                "DERIV_API_TOKEN manquant ou placeholder : le streaming ticks "
                "live est indisponible → le bot fonctionne en mode HISTORIQUE "
                "(rafraîchissement ticks_history à chaque cycle). Configure le "
                "token API pour le temps réel complet."
            )

        await self._telegram(
            "🟢 *Bot Deriv démarré*\n"
            f"Symbole : {self.SYMBOL}\n"
            f"Capital : {self.risk.capital_actuel:.2f}$\n"
            f"Stake auto max : {self.STAKE_AUTO}$\n"
            f"Durée contrat : {self.DURATION}s\n"
            f"Modèles actifs : HMM + XGBoost + LSTM + Kalman + RSI\n"
            f"Règle : trade seulement si 4/5 modèles d'accord"
            + ("" if token_ok else "\n\n⚠️ Mode HISTORIQUE (token API manquant)")
        )

        try:
            await asyncio.gather(
                self._collect_loop(),
                self._analyse_loop(),
            )
        finally:
            await self.collector.close()
            if self.conn is not None:
                await self.conn.close()

    def stop(self):
        self._stop = True

    # ── Session de trading (UTC) pour le journal ────────────────────
    @staticmethod
    def _get_session() -> str:
        """Identifie la session actuelle (london / overlap / newyork / asian)."""
        import pandas as pd

        h = pd.Timestamp.now(tz="UTC").hour
        if 7 <= h < 10:
            return "london"
        if 12 <= h < 15:
            return "overlap"      # London + New York
        if 15 <= h < 20:
            return "newyork"
        return "asian"

    @staticmethod
    def _model_vote(detail: dict, name: str) -> float:
        """Vote d'un modèle (0.5 neutre si absent) — jamais de KeyError."""
        try:
            return float(detail.get(name, {}).get("vote", 0.5))
        except (TypeError, ValueError, AttributeError):
            return 0.5

    # ── Journalisation d'un setup validé (ensemble 4/5 + stratégie) ─
    async def _journaliser_setup(self, df, result: dict) -> None:
        """Loggue le trade dans le journal SQLite (no-op si indisponible)."""
        if self.journal is None:
            return
        try:
            import pandas as pd

            from deriv.ensemble_predictor import build_features

            detail = result.get("detail", {})
            hmm = detail.get("HMM", {})
            signal = self.strategy.analyse(
                df,
                {"regime": hmm.get("regime", "Range"),
                 "confiance": float(hmm.get("confiance", 0.0))},
            )
            strat_phase = signal.phase if signal else "ENSEMBLE"
            rationale = (
                signal.rationale if signal
                else f"Ensemble seul : {result['signal']} "
                     f"{result['buy_votes'] if result['signal'] == 'BUY' else result['sell_votes']}/7"
            )

            direction = "long" if result["signal"] == "BUY" else "short"
            ts = pd.Timestamp.now(tz="UTC")
            trade_id = f"{ts.strftime('%Y%m%d_%H%M')}_{self.SYMBOL}"
            last_close = float(df["close"].iloc[-1])

            feat = build_features(df)
            row = feat.iloc[-1] if len(feat) else None

            def _f(name: str, default: float = 0.0) -> float:
                try:
                    v = float(row[name]) if row is not None else default
                    return v if v == v else default  # NaN → défaut
                except (KeyError, TypeError, ValueError):
                    return default

            entry = JournalEntry(
                trade_id=trade_id,
                timestamp_entry=ts.isoformat(),
                symbol=self.SYMBOL,
                direction=direction,
                entry_price=float(signal.entry_price) if signal else last_close,
                stop_loss=float(signal.stop_loss) if signal else 0.0,
                take_profit=float(signal.take_profit) if signal else 0.0,
                risk_reward=float(signal.risk_reward) if signal else 0.0,
                stake=self.STAKE_AUTO,
                regime=str(hmm.get("regime", "Range")),
                hmm_confidence=float(hmm.get("confiance", 0.0)),
                adx=_f("adx_14", 0.0),
                rsi=_f("rsi_14", 50.0),
                atr=_f("atr_14", 0.0),
                ema_fast=last_close,
                ema_slow=last_close,
                volume_ratio=1.0,
                session=self._get_session(),
                vote_hmm=self._model_vote(detail, "HMM"),
                vote_xgboost=self._model_vote(detail, "XGBoost"),
                vote_lstm=self._model_vote(detail, "LSTM"),
                vote_kalman=self._model_vote(detail, "Kalman"),
                vote_rsi=self._model_vote(detail, "RSI"),
                vote_trend=self._model_vote(detail, "TrendStrength"),
                vote_momentum=self._model_vote(detail, "Momentum"),
                ensemble_votes=int(
                    result["buy_votes"] if result["signal"] == "BUY"
                    else result["sell_votes"]),
                ensemble_signal=result["signal"],
                strategy_phase=strat_phase,
                pullback_candles=int(signal.pullback_candles) if signal else 0,
                rationale=rationale,
                spread_cost_pct=(
                    float(signal.spread_cost_pct)
                    if (self.boom_crash and signal and signal.spread_cost_pct)
                    else 0.2 if self.boom_crash else 0.0
                ),
            )
            self.journal.log_entry(entry)
            self.open_trades[trade_id] = {"entry": entry}
            logger.info(f"Trade journalisé : {trade_id} | {rationale[:100]}")
        except Exception:
            logger.warning("Journalisation ignorée (erreur non bloquante)",
                           exc_info=True)

    # ── Boucles principales ─────────────────────────────────────────
    async def _collect_loop(self):
        """Collecte les ticks en continu (backoff sur erreurs récurrentes).

        Sans token API, l'abonnement live échoue (InvalidSymbol) : le bot reste
        opérationnel via le mode historique de _analyse_loop, mais on évite de
        marteler le serveur avec un backoff exponentiel borné à 30s.
        """
        backoff = 5
        while not self._stop:
            try:
                await self.collector.subscribe_ticks()
                backoff = 5
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                hint = ""
                if "InvalidSymbol" in str(exc):
                    hint = (" — un token API (DERIV_API_TOKEN) est requis pour "
                            "le streaming live ; le mode historique continue de "
                            "tourner tant que tu l'analyses via ticks_history")
                logger.error(f"Flux de ticks interrompu : {exc}{hint}")
            if self._stop:
                break
            logger.info(f"Reconnexion au flux de ticks dans {backoff}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _analyse_loop(self):
        """Analyse toutes les CYCLE_SEC secondes et exécute si 4/5 votent.

        Si le streaming ticks n'a rien produit (token absent ou coupure),
        le collecteur bascule en mode historique rafraîchi : on re-télécharge
        les DERNIÈRES bougies fermées via ticks_history (1 requête/cycle).
        """
        while not self._stop:
            await asyncio.sleep(self.CYCLE_SEC)

            df = self.collector.get_dataframe()
            if df.empty:
                # Mode secours : historique frais (bougies récentes fermées)
                try:
                    fresh = await self.collector.get_candle_history(
                        min(500, 60 * 24)
                    )
                    if fresh.empty:
                        logger.info("Historique vide — attente du prochain cycle...")
                        continue
                    # Recompose à partir des bougies serveur fraîches
                    self.collector.candles = [
                        {
                            "timestamp": int(ts.timestamp()),
                            "open": float(r.open), "high": float(r.high),
                            "low": float(r.low), "close": float(r.close),
                            "volume": float(r.volume),
                        }
                        for ts, r in fresh.iterrows()
                    ][-500:]
                    df = self.collector.get_dataframe()
                    logger.info(
                        f"Mode historique : {len(df)} bougies fraîches chargées"
                    )
                except Exception as exc:
                    logger.warning(f"Rafraîchissement historique impossible : {exc}")
                    continue

            await self._run_cycle(df)

    # ── V2 : features online depuis JournalEntry (boucle fermée) ────
    @staticmethod
    def _online_features(entry) -> dict:
        """Construit le dict de features online depuis une JournalEntry."""
        import numpy as _np
        import pandas as _pd

        hour_sin = hour_cos = 0.0
        day_of_week = 0.0
        try:
            ts = _pd.Timestamp(entry.timestamp_entry)
            hour_sin = float(_np.sin(2 * _np.pi * ts.hour / 24))
            hour_cos = float(_np.cos(2 * _np.pi * ts.hour / 24))
            day_of_week = float(ts.dayofweek) / 6.0
        except Exception as exc:
            logger.warning('Online features failed: %s', exc)
        session = (entry.session or "").lower()
        return {
            "hmm_confidence": entry.hmm_confidence,
            "ensemble_votes": entry.ensemble_votes,
            "rsi": entry.rsi,
            "adx": entry.adx,
            "volume_ratio": entry.volume_ratio,
            "atr": entry.atr,
            "risk_reward": entry.risk_reward,
            "session_london": int(session == "london"),
            "session_overlap": int(session == "overlap"),
            "session_newyork": int(session == "newyork"),
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_of_week": day_of_week,
        }

    # ── Cycle complet : analyse → décision → exécution ──────────────
    async def _run_cycle(self, df):
        result = self.ensemble.predict(df)

        logger.info(
            f"Signal: {result['signal']} | "
            f"Votes BUY:{result['buy_votes']} SELL:{result['sell_votes']} "
            f"HOLD:{result['hold_votes']} | "
            f"Confiance: {result['confiance']:.0%}"
        )

        # RÈGLE ABSOLUE : en dessous de 4/5 → HOLD, on ne trade pas.
        if not result["tradeable"]:
            return

        # V2 — FILTRE SESSION : pas de trade hors London/NY (7h-21h UTC).
        # La session asiatique = moins de liquidité, signaux plus bruités.
        tradeable_session, raison_session = session_filter(df.index[-1])
        if not tradeable_session:
            logger.info(f"HOLD (session) : {raison_session}")
            return

        # Vérification risque
        ok, raison = self.risk.check(self.STAKE_AUTO, result["confiance"] * 100)
        if not ok:
            if "CONFIRMATION_REQUISE" in raison:
                await self._telegram(f"⚠️ {raison}")
            else:
                logger.info(f"Bloqué par Risk Manager : {raison}")
            return

        # Alerte Telegram AVANT d'exécuter
        detail = result["detail"]
        msg = (
            f"{'🟢' if result['signal'] == 'BUY' else '🔴'} "
            f"*{result['signal']}* — {self.SYMBOL}\n"
            f"Votes : {result['buy_votes']}✅/"
            f"{result['sell_votes']}❌/{result['hold_votes']}⏸\n\n"
            f"🧠 HMM : {detail['HMM']['regime']}\n"
            f"📊 XGBoost : {detail['XGBoost']['proba']:.0%} UP\n"
            f"🔮 LSTM : {detail['LSTM']['proba']:.0%} UP\n"
            f"📈 Kalman trend : {detail['Kalman']['trend']:+.4f}\n"
            f"💹 RSI : {detail['RSI']['rsi']}\n\n"
            f"💰 Stake : {self.STAKE_AUTO}$ | Durée : {self.DURATION}s"
        )
        await self._telegram(msg)

        # Exécution du contrat
        contract_type = "CALL" if result["signal"] == "BUY" else "PUT"
        try:
            await self._execute_contract(contract_type)
        except Exception as exc:
            logger.error(f"Erreur exécution : {exc}", exc_info=True)
            await self._telegram(f"❌ Erreur d'exécution : {exc}")

        # Journalisation du setup (mission edge) — APRES execution
        await self._journaliser_setup(df, result)

    async def _execute_contract(self, contract_type: str):
        """Place le contrat Deriv et enregistre le résultat côté RiskManager."""
        proposal = await self.conn.client.trading.proposal(
            contract_type=contract_type,
            symbol=self.SYMBOL,
            amount=self.STAKE_AUTO,
            basis="stake",
            duration=self.DURATION,
            duration_unit="s",
            currency="USD",
        )

        buy_resp = await self.conn.client.trading.buy(
            buy=proposal.proposal.id,
            price=proposal.proposal.ask_price,
        )

        contract_id = int(getattr(buy_resp, "contract_id", 0))
        logger.info(
            f"Contrat acheté : {contract_type} {self.SYMBOL} "
            f"{self.STAKE_AUTO}$ / {self.DURATION}s (id={contract_id})"
        )

        # Attend la fin du contrat + marge de sécurité
        await asyncio.sleep(self.DURATION + 5)

        # Récupère le résultat via proposal_open_contract
        pnl = 0.0
        if contract_id:
            contract = await self.conn.client.contract.get(
                contract_id=contract_id
            )
            pnl = float(getattr(contract, "profit", 0.0))

        self.risk.record(TradeResult(
            symbol=self.SYMBOL,
            pnl=pnl,
            contract_type=contract_type,
            stake=self.STAKE_AUTO,
        ))

        # Clôture journal (mission edge) : tous les trades ouverts sans id
        # Deriv fiable sont clôturés avec le PnL réel du contrat.
        if self.journal is not None and self.open_trades:
            for tid in list(self.open_trades):
                try:
                    exit_res = self.journal.log_exit_contract(tid, pnl_dollar=pnl)
                    # V2 — BOUCLE FERMÉE : le modèle online apprend du trade
                    # clôturé immédiatement (pas d'attente du retrain hebdo).
                    entry = self.open_trades[tid].get("entry")
                    if self.online is not None and entry is not None:
                        upd = self.online.update(
                            self._online_features(entry),
                            label=int((exit_res or {}).get("pnl_net_pct", 0) > 0),
                        )
                        if upd.get("drift_detected"):
                            await self._telegram(
                                "⚠️ *CONCEPT DRIFT DÉTECTÉ*\n"
                                f"Accuracy online : {upd['accuracy_running']:.1%} "
                                f"(drift #{upd['n_drifts']} sur "
                                f"{upd['n_samples']} trades)\n"
                                "Le modèle s'adapte automatiquement."
                            )
                except ValueError as exc:
                    logger.warning('Journal close failed: %s', exc)
                except Exception:
                    logger.warning(f"Clôture journal {tid} ignorée",
                                   exc_info=True)
            self.open_trades.clear()
            # Boucle d'amélioration : rapport hebdo dès 10 trades clôturés.
            try:
                if self.learner is not None:
                    rapport = self.learner.analyse_et_recommande()
                    if rapport.get("profitable") is not None and rapport.get(
                            "n_trades_analyses", 0) >= 10:
                        await self._telegram(
                            rapport.get("message_telegram", ""))
            except Exception:
                logger.warning("Analyse journal ignorée", exc_info=True)

        status = "✅ GAGNÉ" if pnl > 0 else "❌ PERDU"
        await self._telegram(
            f"{status} — {self.SYMBOL}\n"
            f"P&L : {'+' if pnl >= 0 else ''}{pnl:.2f}$\n"
            f"Capital actuel : {self.risk.capital_actuel:.2f}$\n"
            f"Winrate jour : {self.risk.stats['winrate_jour']:.1f}%"
        )

    # ── Telegram ────────────────────────────────────────────────────
    async def _telegram(self, msg: str):
        """Delegation vers TelegramNotifier."""
        await self.telegram.send(msg)

async def main() -> None:
    """Point d'entrée : lancer le bot avec le capital du .env."""
    from dotenv import load_dotenv
    load_dotenv()
    bot = DerivBotExecutor()
    await bot.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    asyncio.run(main())
