"""
════════════════════════════════════════════════════════════════════
VALIDATION DRIFT BOOM/CRASH — Mesure de l'edge réel
deriv/validate_boom_crash_drift.py
════════════════════════════════════════════════════════════════════

Objectif : valider (ou invalider) le drift structurel des indices
Boom 500 / Crash 500 sur données RÉELLES Deriv.

Méthode (out-of-sample, pas de fuite) :
  1. Pour chaque spike détecté (mouvement > 0.5% sur 1 bougie),
     on attend POST_SPIKE_BARS bougies.
  2. On simule l'entrée post-spike dans le sens du drift attendu
     (BOOM → SHORT, CRASH → LONG).
  3. Résultat = prix à la bougie suivante (horizon 1) et jusqu'à
     3 bougies, avec SL/TP par ATR.
  4. On applique le coût du spread (0.2% aller-retour) → pnl NET.

Usage : python deriv/validate_boom_crash_drift.py --symbol BOOM500
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deriv.strategies.boom_crash_drift import BoomCrashDriftStrategy

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("BOOM_CRASH_VALIDATION")


def simulate_drift(df: pd.DataFrame, symbol: str,
                   post_bars: int = 2, horizon: int = 3,
                   spread_pct: float = 0.2) -> dict:
    """
    Simule la stratégie drift sur tout l'historique.

    Retourne les stats de PnL net (spread déduit) sur les trades pris.
    """
    strat = BoomCrashDriftStrategy(symbol=symbol)
    strat.POST_SPIKE_BARS = post_bars

    trades = []
    i = 1
    # Passe unique : avance bougie par bougie en gardant l'état de la
    # machine (stateful). On évalue chaque signal sur les prix FUTURS
    # réels (aucune fuite : la fermeture post-signal n'était pas connue
    # au moment du signal).
    while i < len(df):
        window = df.iloc[: i + 1]
        sig = strat.analyse(window)
        if sig is not None:
            entry = float(sig.entry_price)
            sl = float(sig.stop_loss)
            tp = float(sig.take_profit)
            direction = sig.direction.value
            exit_price = entry
            exit_reason = "time"
            for h in range(1, horizon + 1):
                if i + h >= len(df):
                    break
                c = float(df["close"].iloc[i + h])
                high = float(df["high"].iloc[i + h])
                low = float(df["low"].iloc[i + h])
                if direction == "long":
                    if low <= sl and high >= tp:
                        exit_price = tp; exit_reason = "tp"
                        break
                    if low <= sl:
                        exit_price = sl; exit_reason = "sl"
                        break
                    if high >= tp:
                        exit_price = tp; exit_reason = "tp"
                        break
                else:  # short
                    if high >= sl and low <= tp:
                        exit_price = sl; exit_reason = "sl"
                        break
                    if high >= sl:
                        exit_price = sl; exit_reason = "sl"
                        break
                    if low <= tp:
                        exit_price = tp; exit_reason = "tp"
                        break
                exit_price = c
            if direction == "long":
                pnl_brut = (exit_price - entry) / entry * 100
            else:
                pnl_brut = (entry - exit_price) / entry * 100
            pnl_net = pnl_brut - spread_pct
            trades.append({
                "entry": entry, "exit": exit_price, "direction": direction,
                "sl": sl, "tp": tp, "exit_reason": exit_reason,
                "pnl_brut_pct": round(pnl_brut, 3),
                "pnl_net_pct": round(pnl_net, 3),
            })
        i += 1

    if not trades:
        return {"n_trades": 0, "win_rate": 0.0, "message": "Aucun signal"}

    wins = sum(1 for t in trades if t["pnl_net_pct"] > 0)
    wr = wins / len(trades)
    net_sum = sum(t["pnl_net_pct"] for t in trades)
    brut_sum = sum(t["pnl_brut_pct"] for t in trades)

    return {
        "n_trades": len(trades),
        "win_rate_net": round(wr, 4),
        "pnl_net_total_pct": round(net_sum, 2),
        "pnl_brut_total_pct": round(brut_sum, 2),
        "pnl_net_moyen_pct": round(net_sum / len(trades), 3),
        "spread_pct": spread_pct,
        "par_outcome": {
            "win": sum(1 for t in trades if t["pnl_net_pct"] > 0),
            "loss": sum(1 for t in trades if t["pnl_net_pct"] <= 0),
            "time": sum(1 for t in trades if t["exit_reason"] == "time"),
            "sl": sum(1 for t in trades if t["exit_reason"] == "sl"),
            "tp": sum(1 for t in trades if t["exit_reason"] == "tp"),
        },
    }


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BOOM500",
                   help="BOOM500, BOOM1000, CRASH500, CRASH1000...")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--post", type=int, default=2,
                   help="bougies à attendre après le spike")
    p.add_argument("--horizon", type=int, default=3,
                   help="horizon d'évaluation en bougies")
    args = p.parse_args()

    from deriv.data_collector import DerivDataCollector

    print(f"\nChargement de {args.count} bougies {args.symbol} "
          f"(granularité {args.granularity}s)...")
    collector = DerivDataCollector(args.symbol, args.granularity)
    await collector.connect()
    df = await collector.get_candle_history(args.count)
    print(f"Données chargées : {len(df)} bougies\n")

    result = simulate_drift(df, args.symbol,
                            post_bars=args.post, horizon=args.horizon)

    print("=" * 62)
    print(f"VALIDATION DRIFT — {args.symbol}")
    print("=" * 62)
    if result.get("n_trades", 0) == 0:
        print(result.get("message", "Aucun signal"))
        return result
    print(f"Signaux (trades)      : {result['n_trades']}")
    print(f"Sorties TP/SL/time    : {result['par_outcome']}")
    print(f"Win Rate NET          : {result['win_rate_net']:.1%} "
          f"(seuil 55.6% + spread = ~57%)")
    print(f"PnL brut total        : {result['pnl_brut_total_pct']:+.2f}%")
    print(f"PnL net total         : {result['pnl_net_total_pct']:+.2f}% "
          f"(spread {result['spread_pct']}% inclus)")
    print(f"PnL net moyen         : {result['pnl_net_moyen_pct']:+.3f}%")
    print("=" * 62)
    verdict = (
        "✅ EDGE NET CONFIRMÉ → test en démo réelle 1$"
        if result["win_rate_net"] >= 0.556
        else "❌ Pas d'edge net après spread → rester en mode historique"
    )
    print(verdict)
    return result


if __name__ == "__main__":
    asyncio.run(main())
