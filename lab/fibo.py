"""
Couche ENTRÉE : retracements / extensions de Fibonacci sur des swings CAUSAUX.

Pourquoi un zigzag ATR et pas `smartmoneyconcepts.swing_highs_lows` : cette
bibliothèque (MIT, ~700 étoiles) détecte un sommet avec `shift(-n)`, c'est-à-dire
avec les n bougies FUTURES, puis nettoie les swings avec les swings suivants.
Un backtest construit dessus voit l'avenir. Ici, un sommet n'existe qu'à partir
de la bougie où le prix est redescendu de k × ATR sous lui (même principe que
`ta.pivothigh` de TradingView, qui ne renvoie le pivot qu'à sa confirmation).

Setup long (symétrique en short) :
  structure haussière : dernier sommet H1 > sommet précédent H0 et creux L1 > L0,
  jambe L1 → H1 ; le prix corrige dans la zone 38,2 % – 61,8 % de la jambe ;
  déclencheur : clôture au-dessus du plus haut de la bougie précédente ;
  stop : sous L1 (100 % = structure cassée) ; objectif : extension 127,2 %.
  Le setup est annulé si la correction dépasse 78,6 % ou si un nouveau pivot apparaît.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.engine import Signal, atr14


@dataclass
class Swings:
    """Pour chaque bougie t : pivots CONFIRMÉS connus à la clôture de t."""
    h1: np.ndarray   # dernier sommet confirmé
    h0: np.ndarray   # sommet précédent
    l1: np.ndarray   # dernier creux confirmé
    l0: np.ndarray   # creux précédent
    last: np.ndarray  # +1 si le dernier pivot confirmé est un sommet, -1 un creux
    new: np.ndarray   # True si un pivot a été confirmé à cette bougie


def zigzag(df: pd.DataFrame, k: float = 2.0, atr: np.ndarray | None = None) -> Swings:
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    atr = atr14(df) if atr is None else atr
    n = len(df)
    H1, H0, L1, L0 = (np.full(n, np.nan) for _ in range(4))
    last, new = np.zeros(n, dtype=int), np.zeros(n, dtype=bool)
    hs, ls = [np.nan, np.nan], [np.nan, np.nan]
    direction, ext = 0, np.nan       # 1 : on suit un plus haut ; -1 : un plus bas
    lp = 0
    for t in range(n):
        if direction == 0:
            direction, ext = 1, h[t]
        elif direction == 1:
            if h[t] > ext:
                ext = h[t]
            elif ext - l[t] >= k * atr[t]:
                hs = [hs[1], ext]
                direction, ext, lp, new[t] = -1, l[t], 1, True
        else:
            if l[t] < ext:
                ext = l[t]
            elif h[t] - ext >= k * atr[t]:
                ls = [ls[1], ext]
                direction, ext, lp, new[t] = 1, h[t], -1, True
        H0[t], H1[t], L0[t], L1[t], last[t] = hs[0], hs[1], ls[0], ls[1], lp
    return Swings(H1, H0, L1, L0, last, new)


def fibo_signals(df: pd.DataFrame, k: float = 2.0, zone=(0.382, 0.618),
                 invalid: float = 0.786, ext: float = 1.272,
                 allow: np.ndarray | None = None) -> dict[int, Signal]:
    """Signaux de retracement. `allow[t]` ∈ {-1, 0, +1, 2} filtre le sens permis
    (2 = les deux) : c'est là que se branchent le régime HMM et la tendance."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    sw = zigzag(df, k)
    n = len(df)
    allow = np.full(n, 2) if allow is None else allow
    sigs: dict[int, Signal] = {}
    deep, used = 0.0, False
    for t in range(1, n):
        if sw.new[t]:
            deep, used = 0.0, False
        if used or np.isnan(sw.h0[t]) or np.isnan(sw.l0[t]):
            continue
        H1, H0, L1, L0 = sw.h1[t], sw.h0[t], sw.l1[t], sw.l0[t]
        if sw.last[t] == 1 and H1 > H0 and L1 > L0 and allow[t] in (1, 2):
            rng = H1 - L1
            deep = max(deep, (H1 - l[t]) / rng)
            if deep > invalid:
                used = True
                continue
            if zone[0] <= deep <= zone[1] and c[t] > h[t - 1] and c[t] < H1:
                sigs[t] = Signal(1, sl_price=L1, tp_price=H1 + (ext - 1) * rng)
                used = True
        elif sw.last[t] == -1 and L1 < L0 and H1 < H0 and allow[t] in (-1, 2):
            rng = H1 - L1
            deep = max(deep, (h[t] - L1) / rng)
            if deep > invalid:
                used = True
                continue
            if zone[0] <= deep <= zone[1] and c[t] < l[t - 1] and c[t] > L1:
                sigs[t] = Signal(-1, sl_price=H1, tp_price=L1 - (ext - 1) * rng)
                used = True
    return sigs
