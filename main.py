"""Point d'entrée : bot de tendance MT5 (défaut) et dashboard.

    python main.py                → bot de tendance : un cycle, en simulation (aucun ordre)
    python main.py trend --loop   → un cycle par jour (7 h UTC), en simulation
    python main.py trend --live   → ordres réels (compte DÉMO tant que MT5_ALLOW_REAL=0)
    python main.py trend --reset  → remet à zéro le coupe-circuit de drawdown
    python main.py trend --paper data/fred → même cycle sur des CSV, sans MT5
    python main.py dashboard      → dashboard Streamlit (journal du bot)

L'ancien bot Deriv (ensemble 7 modèles, drift BOOM/CRASH) n'avait aucun edge mesuré :
il est archivé sous le tag git `archive/deriv-ensemble`.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _dashboard_cmd() -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py")]


def run_dashboard() -> None:
    subprocess.run(_dashboard_cmd(), check=False)


def run_trend(args) -> None:
    if os.getenv("BOT_SKIP_DOTENV") != "1":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    from trend_bot import runner

    if args.paper:
        import pandas as pd

        from trend_bot.brokers.paper import PaperBroker
        from trend_bot.data import FRED, charger_fred
        closes = pd.DataFrame(charger_fred(args.paper, list(FRED), depuis="1985"))
        broker = PaperBroker(closes, state_path=ROOT / "data" / "paper_broker.json")
    else:
        from trend_bot.brokers.mt5 import MT5Broker
        broker = MT5Broker()
    if args.reset:
        runner.reset_killswitch(broker)
        print("Coupe-circuit remis à zéro.")
        return
    runner.check_account(broker, live=args.live)
    marches = list(broker.closes.columns) if args.paper else None
    if args.loop:
        runner.loop(broker, dry_run=not args.live)
    else:
        runner.run_cycle(broker, dry_run=not args.live, marches=marches)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="trading-algo")
    parser.add_argument("target", nargs="?", default="trend", choices=["trend", "dashboard"])
    parser.add_argument("--live", action="store_true",
                        help="Envoie réellement les ordres (sinon simulation)")
    parser.add_argument("--loop", action="store_true", help="Un cycle par jour")
    parser.add_argument("--reset", action="store_true", help="Remise à zéro du coupe-circuit")
    parser.add_argument("--paper", metavar="DOSSIER", help="CSV au lieu de MT5")
    args = parser.parse_args(argv)

    handlers = [logging.StreamHandler()]
    if args.target == "trend":          # journal persistant, lu par le dashboard et la tâche planifiée
        Path("logs").mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(Path("logs") / "trend_bot.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=handlers,
    )

    if args.target == "dashboard":
        run_dashboard()
    else:
        run_trend(args)


if __name__ == "__main__":
    main()
