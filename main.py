"""Point d'entrée : lance Deriv (dry-run) et le dashboard Streamlit."""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_dashboard() -> None:
    app = ROOT / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


async def run_deriv(dry_run: bool = True) -> None:
    from deriv.bot import DerivBot

    await DerivBot().start(dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="trading-algo")
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["all", "deriv", "dashboard"],
    )
    parser.add_argument("--live", action="store_true", help="Désactive le dry-run Deriv")
    args = parser.parse_args()

    if args.target in {"deriv", "all"}:
        asyncio.run(run_deriv(dry_run=not args.live))
    if args.target in {"dashboard", "all"}:
        run_dashboard()


if __name__ == "__main__":
    main()
