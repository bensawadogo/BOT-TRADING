"""
═══════════════════════════════════════════════════════════════════
RÉ-ENTRAÎNEMENT HEBDOMADAIRE AUTOMATIQUE
deriv/weekly_retrain.py
═══════════════════════════════════════════════════════════════════

Le bot ne doit jamais trader sur des modèles entraînés il y a des
semaines : les régimes de marché changent. Ce module :

  1. Vérifie la date du dernier entraînement (état JSON persistant)
  2. Si plus de 7 jours → ré-entraîne tous les modèles sur les
     données fraîches Deriv et sauvegarde les nouveaux pickles
  3. Journalise l'historique des ré-entraînements (audit)

Intégré au bot (bot_executor.start()) : au démarrage, si le délai est
dépassé → ré-entraînement automatique, non-bloquant en cas d'erreur.

Usage :
  python deriv/weekly_retrain.py --status          # état actuel
  python deriv/weekly_retrain.py --force           # force le retrain
  python deriv/weekly_retrain.py --daemon          # boucle horaire
"""
from __future__ import annotations
from deriv.constants import Config

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Iterator

try:
    from filelock import FileLock
except ImportError:
    FileLock = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("RETRAIN")

if FileLock is None:
    logger.warning("filelock non installé — verrous inter-processus désactivés")

# Résolution robuste de la racine (fonctionne en script ET en module)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_STATE_FILE = os.path.join(_BASE_DIR, "models", ".retrain_state.json")
MIN_CANDLES_RETRAIN = 300


class RetrainScheduler:
    """
    Planificateur de ré-entraînement hebdomadaire.

    État persistant (JSON) : {"last_train": ISO, "history": [...]}
    """

    def __init__(
        self,
        interval_days: float = 7.0,
        state_file: str | None = None,
        symbol: str | None = None,
        granularity: int | None = None,
        count: int = 1500,
    ):
        self.interval_days = float(interval_days)
        self.state_file = state_file or DEFAULT_STATE_FILE
        self.symbol = symbol or Config.SYMBOL
        self.granularity = int(
            granularity or Config.GRANULARITY
        )
        self.count = int(count)

    # ── État persistant ──────────────────────────────────────────────
    def _lock_path(self) -> str:
        return self.state_file + ".lock"

    def _lock(self):
        if FileLock is None:
            from contextlib import nullcontext
            return nullcontext()
        state_dir = os.path.dirname(os.path.abspath(self.state_file))
        os.makedirs(state_dir, exist_ok=True)
        return FileLock(self._lock_path())

    def _load_state_unlocked(self) -> dict:
        if not os.path.exists(self.state_file):
            return {}
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"État retrain illisible : {exc}")
            return {}

    def load_state(self) -> dict:
        with self._lock():
            return self._load_state_unlocked()

    def _save_state_unlocked(self, state: dict) -> None:
        state_dir = os.path.dirname(os.path.abspath(self.state_file))
        os.makedirs(state_dir, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{os.path.basename(self.state_file)}.",
            suffix=".tmp",
            dir=state_dir,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.state_file)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise

    def _save_state(self, state: dict) -> None:
        with self._lock():
            self._save_state_unlocked(state)

    # ── Décision de ré-entraînement ─────────────────────────────────
    def needs_retrain(self, now: datetime | None = None) -> bool:
        """True si jamais entraîné OU si l'intervalle est dépassé."""
        state = self.load_state()
        last = state.get("last_train")
        if not last:
            return True
        try:
            last_dt = datetime.fromisoformat(last)
        except (ValueError, TypeError):
            return True
        now = now or datetime.now()
        return (now - last_dt) >= timedelta(days=self.interval_days)

    def mark_trained(
        self, now: datetime | None = None, meta: dict | None = None
    ) -> dict:
        """Enregistre l'entraînement et l'ajoute à l'historique d'audit."""
        now = now or datetime.now()
        state = self.load_state()
        state["last_train"] = now.isoformat()
        history = state.get("history", [])
        entry = {"timestamp": now.isoformat(), **(meta or {})}
        history.append(entry)
        state["history"] = history[-50:]   # borné : 50 dernières entrées
        self._save_state(state)
        return state

    def history(self) -> list[dict]:
        return self.load_state().get("history", [])

    # ── Ré-entraînement effectif ─────────────────────────────────────
    async def retrain_async(self, df=None) -> dict:
        """
        Ré-entraîne tous les modèles sur `df` (ou sur des données fraîches
        Deriv si df est None). Retourne un rapport dict (jamais d'exception).
        """
        if df is None:
            try:
                from deriv.data_collector import DerivDataCollector

                collector = DerivDataCollector(self.symbol, self.granularity)
                await collector.connect()
                df = await collector.get_candle_history(self.count)
                await collector.close()
            except Exception as exc:
                logger.error(f"Chargement données impossible : {exc}")
                return {"ok": False, "erreur": f"chargement: {exc}"}

        if df is None or len(df) < 300:
            return {"ok": False,
                    "erreur": f"pas assez de bougies ({0 if df is None else len(df)})"}

        try:
            from deriv.ensemble_predictor import EnsemblePredictor

            ensemble = EnsemblePredictor()
            ensemble.train_all(df)
        except Exception as exc:
            logger.error(f"Entraînement échoué : {exc}")
            return {"ok": False, "erreur": f"entrainement: {exc}"}

        self.mark_trained(meta={
            "symbol": self.symbol,
            "granularity": self.granularity,
            "n_bougies": len(df),
        })
        logger.info(f"Ré-entraînement OK : {len(df)} bougies {self.symbol}")
        return {"ok": True, "n_bougies": len(df), "symbol": self.symbol}

    def retrain(self, df=None) -> dict:
        """Version synchrone (scripts CLI)."""
        import asyncio

        return asyncio.run(self.retrain_async(df=df))


def _configure_console_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main():
    _configure_console_utf8()
    p = argparse.ArgumentParser(description="Ré-entraînement hebdomadaire")
    p.add_argument("--interval-days", type=float, default=7.0)
    p.add_argument("--symbol", default=None)
    p.add_argument("--granularity", type=int, default=None)
    p.add_argument("--count", type=int, default=1500)
    p.add_argument("--status", action="store_true",
                   help="Affiche l'état sans ré-entraîner")
    p.add_argument("--force", action="store_true",
                   help="Force le ré-entraînement immédiat")
    p.add_argument("--daemon", action="store_true",
                   help="Vérifie toutes les heures (tourne en continu)")
    args = p.parse_args()

    sched = RetrainScheduler(
        interval_days=args.interval_days,
        symbol=args.symbol,
        granularity=args.granularity,
        count=args.count,
    )

    if args.status:
        state = sched.load_state()
        print(f"Dernier entraînement : {state.get('last_train', 'JAMAIS')}")
        print(f"Ré-entraînement requis : {'OUI' if sched.needs_retrain() else 'non'}")
        print(f"Historique : {len(sched.history())} entrée(s)")
        return 0

    if args.daemon:
        logger.info("Mode daemon : vérification toutes les heures (Ctrl+C pour stop)")
        import asyncio
        import time

        while True:
            if sched.needs_retrain():
                asyncio.run(sched.retrain_async())
            time.sleep(3600)

    if args.force or sched.needs_retrain():
        rep = sched.retrain()
        print(("OK " + str(rep)) if rep.get("ok") else ("ECHEC " + str(rep)))
        return 0 if rep.get("ok") else 1

    print("Pas de ré-entraînement requis (dernier < 7 jours).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



