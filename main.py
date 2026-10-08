"""Point d'entrée : lance le bot Deriv (dry-run par défaut) et/ou le dashboard.

    python main.py deriv          → bot complet en mode papier (aucun achat)
    python main.py deriv --live   → achète réellement les contrats
                                    (compte DEMO tant que DERIV_ALLOW_REAL=0)
    python main.py dashboard      → dashboard Streamlit seul
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="trading-algo")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "deriv", "dashboard"],
    )
    parser.add_argument("--live", action="store_true",
                        help="Achète réellement les contrats (sinon dry-run)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.target == "dashboard":
        run_dashboard()
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
