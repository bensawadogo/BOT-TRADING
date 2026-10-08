"""Validation anti-leakage sur données réelles H4 (frxEURUSD)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deriv.data_collector import DerivDataCollector
from deriv.leakage_detector import run_leakage_check


async def main():
    c = DerivDataCollector("frxEURUSD", 14400)
    await c.connect()
    try:
        df = await c.get_candle_history_full(1000)
        print(f"{len(df)} bougies H4 chargees")
        run_leakage_check(df)
    finally:
        await c.close()


asyncio.run(main())
