"""
Moteur de backtest bar-par-bar, sans regard vers le futur.

- Le signal est calculé à la CLÔTURE de la bougie i (données <= i seulement).
- L'entrée se fait à l'OUVERTURE de la bougie i+1.
- Sorties : stop / objectif intra-bougie (si les deux sont touchés dans la même
  bougie, on compte le STOP — hypothèse prudente), durée max, signal de sortie,
  ou table ROI façon freqtrade.
- Coût : commission réelle Deriv (fraction du notionnel), déduite à chaque trade.
- Une seule position à la fois.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Signal:
    side: int                     # +1 achat, -1 vente
    sl_atr: float | None = None   # stop = sl_atr × ATR(14) de la bougie du signal
    tp_atr: float | None = None   # objectif = tp_atr × ATR(14)
    sl_price: float | None = None  # niveaux absolus (Fibonacci) : prioritaires sur l'ATR
    tp_price: float | None = None


@dataclass
class ExitRules:
    max_hold: int = 48                       # bougies
    exit_signal: np.ndarray | None = None    # True → sortie à la clôture
    roi: dict[int, float] | None = None      # {bougies écoulées: rendement min}
    stoploss_pct: float | None = None        # stop en % (si pas de sl_dist)


@dataclass
class Trade:
    entry_i: int
    exit_i: int
    side: int
    entry: float
    exit: float
    reason: str
    ret: float = 0.0     # rendement net (fraction du notionnel)


@dataclass
class Result:
    trades: list[Trade] = field(default_factory=list)

    @property
    def rets(self) -> np.ndarray:
        return np.array([t.ret for t in self.trades])

    def stats(self, n_bars: int | None = None) -> dict:
        r = self.rets
        if r.size == 0:
            return {"n": 0}
        gains, pertes = r[r > 0].sum(), -r[r < 0].sum()
        equity = np.cumsum(r)
        dd = float((np.maximum.accumulate(equity) - equity).max())
        return {
            "n": int(r.size),
            "wr": float((r > 0).mean()),
            "moy_bps": float(r.mean() * 1e4),
            "pf": float(gains / pertes) if pertes > 0 else float("inf"),
            "total_pct": float(r.sum() * 100),
            "max_dd_pct": dd * 100,
        }


def atr14(df: pd.DataFrame) -> np.ndarray:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean().to_numpy(float)


def run(df: pd.DataFrame, signals: dict[int, Signal], rules: ExitRules,
        cost: float, start: int = 0, end: int | None = None,
        atr: np.ndarray | None = None) -> Result:
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = atr14(df) if atr is None else atr
    end = len(df) if end is None else end
    res, i = Result(), start
    sig_idx = sorted(k for k in signals if start <= k < end - 1)
    pos = 0
    while pos < len(sig_idx):
        i = sig_idx[pos]
        s = signals[i]
        e_i = i + 1
        entry = o[e_i]
        sl = entry - s.side * s.sl_atr * atr[i] if s.sl_atr else (
            entry * (1 - s.side * rules.stoploss_pct) if rules.stoploss_pct else None)
        tp = entry + s.side * s.tp_atr * atr[i] if s.tp_atr else None
        if s.sl_price is not None:
            sl = s.sl_price
        if s.tp_price is not None:
            tp = s.tp_price
        if (sl is not None and s.side * (entry - sl) <= 0) or (
                tp is not None and s.side * (tp - entry) <= 0):
            pos += 1          # l'ouverture a déjà franchi un niveau : setup caduc
            continue
        exit_px, reason, j = None, "", e_i
        for j in range(e_i, min(e_i + rules.max_hold, end)):
            if sl is not None and ((s.side > 0 and l[j] <= sl) or (s.side < 0 and h[j] >= sl)):
                exit_px, reason = (min(sl, o[j]) if s.side > 0 else max(sl, o[j])), "stop"
                break
            if tp is not None and ((s.side > 0 and h[j] >= tp) or (s.side < 0 and l[j] <= tp)):
                exit_px, reason = (max(tp, o[j]) if s.side > 0 else min(tp, o[j])), "objectif"
                break
            if rules.roi:
                gain = s.side * (c[j] / entry - 1)
                seuil = [v for k, v in sorted(rules.roi.items()) if j - e_i >= k]
                if seuil and gain >= seuil[-1]:
                    exit_px, reason = c[j], "roi"
                    break
            if rules.exit_signal is not None and rules.exit_signal[j]:
                exit_px, reason = c[j], "signal"
                break
        if exit_px is None:
            exit_px, reason = c[j], "durée"
        t = Trade(e_i, j, s.side, entry, exit_px, reason)
        t.ret = s.side * (exit_px / entry - 1) - cost
        res.trades.append(t)
        # prochaine entrée possible après la sortie
        while pos < len(sig_idx) and sig_idx[pos] <= j:
            pos += 1
    return res


def random_baseline(df: pd.DataFrame, signals: dict[int, Signal], rules: ExitRules,
                    cost: float, start: int, end: int, n_sims: int = 200,
                    seed: int = 0) -> np.ndarray:
    """Mêmes sorties et même nombre / sens de signaux, mais placés au hasard."""
    rng = np.random.default_rng(seed)
    atr = atr14(df)
    pool = [k for k in signals if start <= k < end - 1]
    if not pool:
        return np.array([])
    gabarits = [signals[k] for k in pool]
    candidats = np.arange(max(start, 50), end - 1)
    moyennes = []
    for _ in range(n_sims):
        pos = rng.choice(candidats, size=len(pool), replace=False)
        fake = {int(p): gabarits[rng.integers(len(gabarits))] for p in pos}
        r = run(df, fake, rules, cost, start, end, atr=atr).rets
        if r.size:
            moyennes.append(r.mean())
    return np.array(moyennes)
