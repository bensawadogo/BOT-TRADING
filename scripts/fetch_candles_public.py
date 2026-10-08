"""Télécharge des bougies M1 Deriv via le NOUVEL endpoint public (sans auth).

L'ancien endpoint ws.derivws.com (app_id 1089) a répondu HTTP 520 le 08/10/2026.
Usage : python scripts/fetch_candles_public.py BOOM500 100000 1000 data
(1000 bougies max par requête côté serveur)
"""
import asyncio, json, sys, websockets, pandas as pd
URL = "wss://api.derivws.com/trading/v1/options/ws/public"
async def fetch(symbol, total, gran, chunk):
    rows, end = [], "latest"
    async with websockets.connect(URL, open_timeout=30, max_size=2**25) as ws:
        while len(rows) < total:
            req = {"ticks_history": symbol, "count": min(chunk, total - len(rows)), "end": end,
                   "style": "candles", "granularity": gran}
            await ws.send(json.dumps(req))
            r = json.loads(await ws.recv())
            if "error" in r:
                print("ERREUR", r["error"]); break
            c = r.get("candles", [])
            if not c: break
            rows = c + rows
            end = str(c[0]["epoch"] - 1)
            print(f"{len(rows)} bougies (début {pd.to_datetime(c[0]['epoch'], unit='s')})", flush=True)
            await asyncio.sleep(0.3)
    df = pd.DataFrame(rows).drop_duplicates("epoch").sort_values("epoch")
    df["timestamp"] = pd.to_datetime(df["epoch"], unit="s")
    return df.set_index("timestamp")[["open", "high", "low", "close"]].astype(float)
sym, total = sys.argv[1], int(sys.argv[2])
df = asyncio.run(fetch(sym, total, 60, int(sys.argv[3]) if len(sys.argv) > 3 else 5000))
df.to_csv(f"{sys.argv[4] if len(sys.argv)>4 else '.'}/{sym}_M1.csv")
print("TOTAL", len(df), df.index.min(), "→", df.index.max())
