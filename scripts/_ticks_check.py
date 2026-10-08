"""Test rapide du streaming Deriv (ticks/candles) + liste active_symbols (sans token)."""
import asyncio
import sys

sys.path.insert(0, r"c:\BOT-TRADING")
from deriv.client import DerivConnection


async def test_candle_stream(conn):
    """Tente un abonnement aux bougies R_75 (ticks_history + subscribe=1)
    via le moteur de requêtes brut du SDK."""
    try:
        # Enregistre le payload subscribe=1 directement
        sub = await asyncio.wait_for(
            conn.client.request_engine.subscribe(
                {"ticks_history": "R_75", "granularity": 60,
                 "style": "candles", "subscribe": 1},
            ),
            timeout=15,
        )
        got = 0
        for _ in range(3):
            msg = await asyncio.wait_for(sub.get(), timeout=15)
            print(f"CANDLE_MSG keys={list(msg.keys())[:4]}")
            got += 1
        print("CANDLE_STREAM_OK", got)
    except Exception as exc:
        print(f"CANDLE_STREAM_FAIL {type(exc).__name__}: {str(exc)[:140]}")


async def main():
    conn = DerivConnection(app_id="1089")
    try:
        await asyncio.wait_for(conn.connect(), timeout=20)
        print("CONNECTED")

        # 1) Streaming ticks frxEURUSD
        try:
            sub = await asyncio.wait_for(
                conn.client.market.subscribe_ticks("frxEURUSD"), timeout=15
            )
            got = 0
            async for tick in sub:
                print(f"TICK {tick.symbol} {tick.quote} @ {tick.epoch}")
                got += 1
                if got >= 3:
                    break
            print("STREAM_TICKS_FRX_OK", got)
        except Exception as exc:
            print(f"STREAM_TICKS_FRX_FAIL {type(exc).__name__}: {str(exc)[:140]}")

        # 2) Streaming candles R_75 (fallback potentiel)
        await test_candle_stream(conn)

        # 3) Liste active_symbols (sans token)
        try:
            symbols = await asyncio.wait_for(
                conn.client.market.active_symbols(product_type="basic"),
                timeout=15,
            )
            items = list(getattr(symbols, "active_symbols", []))
            print(f"ACTIVE_SYMBOLS_COUNT={len(items)}")
            for s in items[:20]:
                print("  ", getattr(s, "symbol", "?"), "-",
                      str(getattr(s, "display_name", "?"))[:40])
        except Exception as exc:
            print(f"ACTIVE_SYMBOLS_FAIL {type(exc).__name__}: {str(exc)[:140]}")
    finally:
        try:
            await conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())