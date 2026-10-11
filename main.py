"""Point d'entrée : lance le bot Deriv (dry-run par défaut) et/ou le dashboard.

    python main.py deriv          → bot complet en mode papier (aucun achat)
    python main.py deriv --live   → achète réellement les contrats
                                    (compte DEMO tant que DERIV_ALLOW_REAL=0)
    python main.py dashboard      → dashboard Streamlit seul
    python main.py trend          → bot de tendance MT5 : un cycle, en simulation
    python main.py trend --loop   → un cycle par jour (7 h UTC), en simulation
    python main.py trend --live   → ordres réels (compte DÉMO tant que MT5_ALLOW_REAL=0)
    python main.py trend --reset  → remet à zéro le coupe-circuit de drawdown
    python main.py trend --paper data/fred → même cycle sur des CSV, sans MT5
    python main.py                → dashboard en arrière-plan + bot
"""

from __future__ import annotations

import argparse
import asyncio
import logging
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


async def run_deriv(dry_run: bool = True) -> None:
    # Le vrai bot : collecte → ensemble/stratégie → risque → exécution.
    # (deriv.bot.DerivBot ne faisait qu'un ping puis s'arrêtait.)
    from deriv.bot_executor import main as executor_main

    await executor_main(dry_run=dry_run)


def run_trend(args) -> None:
    import os

    if os.getenv("BOT_SKIP_DOTENV") != "1":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    from trend_bot import runner

    if args.paper:
        import pandas as pd

        from lab.daily import charger_fred
        from trend_bot.brokers.paper import PaperBroker
        from lab.run_systeme import FRED
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
    marches = None
    if args.paper:
        marches = list(broker.closes.columns)
    if args.loop:
        runner.loop(broker, dry_run=not args.live)
    else:
        runner.run_cycle(broker, dry_run=not args.live, marches=marches)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="trading-algo")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "deriv", "dashboard", "trend"],
    )
    parser.add_argument("--live", action="store_true",
                        help="Achète réellement les contrats (sinon dry-run)")
    parser.add_argument("--loop", action="store_true", help="trend : un cycle par jour")
    parser.add_argument("--reset", action="store_true", help="trend : reset du coupe-circuit")
    parser.add_argument("--paper", metavar="DOSSIER", help="trend : CSV au lieu de MT5")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.target == "dashboard":
        run_dashboard()
        return
    if args.target == "trend":
        run_trend(args)
        return

    # Le bot tourne en continu : le dashboard est lancé à côté, pas après.
    dashboard = subprocess.Popen(_dashboard_cmd()) if args.target == "all" else None
    try:
        asyncio.run(run_deriv(dry_run=not args.live))
    except KeyboardInterrupt:
        pass
    finally:
        if dashboard is not None:
            dashboard.terminate()


if __name__ == "__main__":
    main()
