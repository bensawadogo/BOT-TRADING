"""Coûts réels relevés sur le terminal MT5 (lecture seule : aucun ordre).

    python scripts/couts_mt5.py tendance   → spreads à 7 h UTC, swaps, lot minimum
                                             par marché du bot → data/couts_mt5.json
    python scripts/couts_mt5.py scalping   → ce que coûte le spread face aux mouvements
                                             de 1 à 60 minutes, et un scalpeur naïf

Spread mesuré sur les ticks (ask − bid), pas sur le champ `spread` des bougies
(minimum de la minute, trop optimiste). Heures en UTC (serveur Deriv = GMT+0).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trend_bot import config  # noqa: E402

SCALPING = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US Tech 100", "US SP 500", "BTCUSD",
            "Volatility 75 Index", "Volatility 10 Index", "Boom 500 Index", "Crash 500 Index"]
HORIZONS_MIN = (1, 5, 15, 60)


def _mt5():
    import MetaTrader5 as mt5
    if not mt5.initialize():
        raise SystemExit(f"MT5 indisponible : {mt5.last_error()}")
    return mt5


def _jours_recents(n: int, semaine: bool) -> list[datetime]:
    j, out = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0), []
    while len(out) < n:
        j -= timedelta(days=1)
        if not semaine or j.weekday() < 5:
            out.append(j)
    return out


def spread_relatif(mt5, sym: str, heures: tuple[int, ...], jours: list[datetime]) -> float:
    """Médiane de (ask − bid) / milieu sur des fenêtres d'une heure (fraction du prix)."""
    vals = []
    for j in jours:
        for h in heures:
            tk = mt5.copy_ticks_range(sym, j + timedelta(hours=h), j + timedelta(hours=h + 1),
                                      mt5.COPY_TICKS_INFO)
            if tk is not None and len(tk):
                d = pd.DataFrame(tk)
                vals.append(((d.ask - d.bid) / ((d.ask + d.bid) / 2)).median())
    return float(np.median(vals)) if vals else float("nan")


def swap_annuel(mt5, info, lot_value: float) -> tuple[float, float]:
    """Swaps long / court en fraction du notionnel par an (négatif = coût)."""
    m = info.swap_mode
    if m == mt5.SYMBOL_SWAP_MODE_DISABLED:
        return 0.0, 0.0
    if m == mt5.SYMBOL_SWAP_MODE_POINTS:
        par_point = info.point * info.trade_tick_value / info.trade_tick_size
        nuit = (info.swap_long * par_point, info.swap_short * par_point)
    elif m == mt5.SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT:
        nuit = (info.swap_long, info.swap_short)
    elif m in (mt5.SYMBOL_SWAP_MODE_INTEREST_CURRENT, mt5.SYMBOL_SWAP_MODE_INTEREST_OPEN):
        return info.swap_long / 100, info.swap_short / 100
    else:
        return float("nan"), float("nan")
    return nuit[0] * 365 / lot_value, nuit[1] * 365 / lot_value


def tendance() -> None:
    mt5 = _mt5()
    jours = _jours_recents(5, semaine=True)
    noms = {s.name for s in mt5.symbols_get()}
    out = {}
    print(f"{'marché':8} {'symbole':14} {'panier':6} {'½ spread':>9} {'swap L':>7} {'swap C':>7} "
          f"{'surcoût':>7} {'lot min':>10}")
    for m, cands in config.ALIASES.items():
        sym = next((c for c in cands if c in noms), None)
        if sym is None:
            print(f"{m:8} absent")
            continue
        mt5.symbol_select(sym, True)
        info, tick = mt5.symbol_info(sym), mt5.symbol_info_tick(sym)
        prix = (tick.bid + tick.ask) / 2 if tick and tick.bid > 0 else \
            float(mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 1)["close"][-1])
        lot_value = prix * info.trade_tick_value / info.trade_tick_size
        demi = spread_relatif(mt5, sym, (7,), jours) / 2
        sl, ss = swap_annuel(mt5, info, lot_value)
        surcout = -(sl + ss) / 2          # marge du courtier sur le différentiel de taux
        out[m] = {"symbole": sym, "panier": m in config.UNIVERSE, "cout": demi,
                  "swap_long": sl, "swap_short": ss, "surcout_swap": surcout,
                  "notionnel_min": info.volume_min * lot_value}
        print(f"{m:8} {sym:14} {'oui' if m in config.UNIVERSE else 'non':6} {demi*1e4:7.2f}bp "
              f"{sl:7.1%} {ss:7.1%} {surcout:7.1%} {info.volume_min * lot_value:10,.0f}")
    dest = ROOT / "data" / "couts_mt5.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps({"releve_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                                "marches": out}, indent=2))
    print(f"\n→ {dest} (½ spread = coût par unité de rotation ; surcoût = −(swap L + swap C)/2)")


def scalping() -> None:
    mt5 = _mt5()
    noms = {s.name for s in mt5.symbols_get()}
    print("Spread : médiane des ticks à 8, 13 et 16 h UTC sur 5 jours. Mouvement : médiane de "
          "|clôture(t+h) − clôture(t)| sur ~20 000 bougies M1.\n"
          "Réussite nécessaire = 0,5 + spread / (2 × mouvement) pour un objectif et un stop "
          "placés à un mouvement médian.\n")
    print(f"{'symbole':20} {'spread':>7} | " + " | ".join(f"{h:>2} min : coût/mvt  réussite"
                                                       for h in HORIZONS_MIN))
    naif = []
    for sym in SCALPING:
        if sym not in noms:
            print(f"{sym:20} absent")
            continue
        mt5.symbol_select(sym, True)
        synth = mt5.symbol_info(sym).path.split("\\")[0].endswith("Indices") and "Stock" not in \
            mt5.symbol_info(sym).path
        sp = spread_relatif(mt5, sym, (8, 13, 16), _jours_recents(5, semaine=not synth))
        r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M1, 1, 20000)
        c = pd.Series(r["close"], index=pd.to_datetime(r["time"], unit="s"))
        cols = []
        for h in HORIZONS_MIN:
            mvt = (np.log(c).diff(h).abs()).median()
            cols.append(f"{sp / mvt:>13.0%}  {0.5 + sp / (2 * mvt):>8.0%}")
        print(f"{sym:20} {sp*1e4:5.2f}bp | " + " | ".join(cols))
        # Scalpeur naïf : à chaque clôture M1, suit (ou contre) la dernière minute, garde h min.
        lr = np.log(c).diff()
        for h in (1, 5):
            fut = np.log(c).shift(-h) - np.log(c)
            for nom, sens in (("suit", 1), ("contre", -1)):
                g = (sens * np.sign(lr) * fut).dropna()
                g = g[np.sign(lr).reindex(g.index) != 0]
                net = g - sp
                naif.append((sym, h, nom, g.mean() * 1e4, net.mean() * 1e4,
                             net.mean() / net.std() * np.sqrt(len(net)), len(net)))
    print("\nScalpeur naïf (une position par minute, spread payé à chaque trade) :")
    print(f"{'symbole':20} {'durée':>5} {'règle':6} {'brut/trade':>10} {'net/trade':>10} {'t net':>7}")
    for sym, h, nom, brut, net, t, n in naif:
        print(f"{sym:20} {h:>3}min {nom:6} {brut:8.2f}bp {net:8.2f}bp {t:7.1f}")


if __name__ == "__main__":
    {"tendance": tendance, "scalping": scalping}[sys.argv[1] if len(sys.argv) > 1 else "tendance"]()
