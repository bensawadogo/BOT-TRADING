"""
Audit d'edge BOOM/CRASH sur données réelles (08/10/2026).

1. Le signal spike bat-il le TAUX DE BASE (n'importe quelle minute) ?
2. Avec les seuls contrats disponibles sur Boom/Crash (Multipliers : pas de
   Rise/Fall), le drift est-il rentable une fois commissions et spikes
   contraires (saut au-delà du stop-out) comptés ?

Usage : python scripts/fetch_candles_public.py BOOM500 100000 1000 data
        python scripts/fetch_candles_public.py CRASH500 100000 1000 data
        python scripts/audit_boom_crash.py data
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deriv.strategies.spike_drift_binary import SpikeDriftBinaryStrategy  # noqa: E402

H = 10
STAKE = 10.0
# Proposals réelles Deriv (mise 10 $) relevées le 08/10/2026
COMM = {100: .09, 150: .14, 200: .19, 300: .28, 400: .38}
STOP = {100: .00891, 150: .00524, 200: .00341, 300: .00191, 400: .00116}


def signaux(df: pd.DataFrame, sym: str) -> np.ndarray:
    st, out = SpikeDriftBinaryStrategy(sym), []
    for i in range(st.min_bars - 1, len(df) - H):
        if st.evaluate(df.iloc[i - st.min_bars + 1:i + 1]):
            out.append(i)
    return np.array(out)


def pnl_multiplier(df, entries, sign, m, h):
    c, hi, lo = df.close.to_numpy(), df.high.to_numpy(), df.low.to_numpy()
    out = []
    for i in entries:
        if i + h >= len(c):
            continue
        e = c[i]
        adv = hi[i + 1:i + h + 1] / e - 1 if sign < 0 else 1 - lo[i + 1:i + h + 1] / e
        hit = np.nonzero(adv >= STOP[m])[0]
        if hit.size:   # spike : saut au-delà du stop-out, perte bornée à la mise
            out.append(-min(STAKE, STAKE * m * adv[hit[0]] + COMM[m]))
        else:
            out.append(STAKE * m * (c[i + h] / e - 1) * sign - COMM[m])
    return np.array(out)


def main(data_dir: str) -> None:
    for sym in ("BOOM500", "CRASH500"):
        df = pd.read_csv(f"{data_dir}/{sym}_M1.csv", index_col=0, parse_dates=True)
        sign = -1 if "BOOM" in sym else 1
        c = df.close.to_numpy()
        fwd = (c[H:] - c[:-H]) * sign
        win = fwd > 0
        base = win[1500:].mean()
        sig = signaux(df, sym)
        w = win[sig]
        p = binomtest(int(w.sum()), len(w), base, alternative="greater").pvalue
        print(f"\n=== {sym} : {len(df)} bougies M1 ===")
        print(f"Taux de base {base:.1%} | après spike {w.mean():.1%} ({len(w)} trades) "
              f"| p(signal > base) = {p:.3f}")
        rnd = np.arange(1500, len(df) - 61, 7)
        for m in (100, 200, 400):
            for h in (5, 10, 30, 60):
                a = pnl_multiplier(df, sig, sign, m, h)
                b = pnl_multiplier(df, rnd, sign, m, h)
                print(f"  x{m:<3} {h:>2} min : après spike {a.mean():+.3f} $/trade "
                      f"| au hasard {b.mean():+.3f} $/trade")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
