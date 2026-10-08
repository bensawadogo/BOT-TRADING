"""Test rapide de connectivité WebSocket Deriv (app_id public 1089, sans token)."""
import asyncio
import sys

sys.path.insert(0, r"c:\BOT-TRADING")
from deriv.client import DerivConnection


async def main():
    conn = DerivConnection(app_id="1089")
    try:
        await asyncio.wait_for(conn.connect(), timeout=20)
        print("DERIV_NETWORK_OK")
    except Exception as exc:
        print(f"DERIV_NETWORK_FAIL {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        try:
            await conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())