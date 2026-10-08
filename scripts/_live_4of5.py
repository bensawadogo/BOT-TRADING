"""Teste le 4/5 sur de vraies données Deriv R_75 (données réelles du broker)."""
import sys; sys.path.insert(0, r"c:\BOT-TRADING")
import asyncio, numpy as np, pandas as pd
from deriv.ensemble_predictor import EnsemblePredictor
from deriv.client import DerivConnection

async def main():
    conn = DerivConnection()
    await conn.connect()

    # Récupère les 500 dernières bougies R_75
    df = await conn.ticks_history(symbol="R_75", count=500, granularity=60, style="candles")
    if df is None or df.empty:
        print("Pas de données R_75")
        return

    print(f"Données R_75: {len(df)} bougies, de {df.index[0]} à {df.index[-1]}")

    ens = EnsemblePredictor()
    tradeable = 0
    total = 0

    # Gliding window: analyse toutes les 10 bougies
    for i in range(100, len(df), 10):
        window = df.iloc[:i]
        res = ens.predict(window)
        total += 1
        if res["tradeable"]:
            tradeable += 1
            print(f"\n*** TRADE à t={i} ***")
            print(f"  Signal: {res['signal']} | buy={res['buy_votes']} sell={res['sell_votes']} hold={res['hold_votes']}")
            for m, d in res["detail"].items():
                print(f"    {m}: vote={d['vote']}")

    print(f"\n=== RÉSUMÉ: {tradeable}/{total} fenêtres tradeable ===")
    if tradeable == 0:
        print("Aucun 4/5 — les modèles ne convergent pas encore")

asyncio.run(main())