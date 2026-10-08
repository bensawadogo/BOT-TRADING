"""
Système complet : RÉGIME (HMM Baum-Welch + Forward + Viterbi, ou modèle à sauts)
+ TENDANCE (momentum temporel) + ENTRÉE (retracement de Fibonacci).

Usage :
  python -m lab.run_systeme daily <dossier_fred>          # 1990-2026, multi-marchés
  python -m lab.run_systeme h1 <csv> <symbole> [...]       # bougies Deriv, coûts réels

Paramètres FIXÉS avant le test (aucune optimisation) : pas de choix a posteriori.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from lab import daily as D
from lab.engine import ExitRules, random_baseline, run
from lab.fibo import fibo_signals
from lab.hmm_causal import walk_forward_hmm, walk_forward_jump

FRED = {  # identifiant: coût par unité de rotation (fraction)
    "DEXUSEU": 1e-4, "DEXUSUK": 1e-4, "DEXJPUS": 1e-4, "DEXUSAL": 1e-4, "DEXCAUS": 1e-4,
    "DEXSZUS": 1e-4, "DEXUSNZ": 2e-4, "DEXMXUS": 3e-4, "DEXNOUS": 3e-4, "DEXSDUS": 3e-4,
    "DCOILWTICO": 5e-4, "DCOILBRENTEU": 5e-4, "NASDAQCOM": 2e-4, "DHHNGSP": 1e-3,
}
PERIODES = {"1990-2007": ("1990", "2007"), "2008-2016": ("2008", "2016"),
            "2017-2026": ("2017", "2026")}


def _ohlc_depuis_cloture(close: pd.Series) -> pd.DataFrame:
    """Clôtures seules : ouverture = clôture précédente, plus haut/bas = clôture."""
    return pd.DataFrame({"open": close.shift(1).fillna(close), "high": close,
                         "low": close, "close": close})


def fibo_positions(close: pd.Series, allow: np.ndarray | None, max_hold: int = 60,
                   k: float = 2.0) -> pd.Series:
    """Position -1/0/+1 décidée à chaque clôture ; stop et objectif jugés à la clôture."""
    df = _ohlc_depuis_cloture(close)
    sigs = fibo_signals(df, k=k, allow=allow)
    c = close.to_numpy(float)
    pos = np.zeros(len(c))
    t = 0
    for i in sorted(sigs):
        if i < t:
            continue
        s = sigs[i]
        j = i
        while j < min(i + max_hold, len(c) - 1):
            pos[j] = s.side
            j += 1
            if s.side * (c[j] - s.sl_price) <= 0 or s.side * (c[j] - s.tp_price) >= 0:
                break
        t = j + 1
    return pd.Series(pos, index=close.index)


def _ligne(nom: str, r: pd.Series) -> str:
    st = D.stats(r)
    if "sharpe" not in st:
        return f"{nom:26} pas assez de données"
    per = []
    for a, b in PERIODES.values():
        s2 = D.stats(r[a:b])
        per.append(f"{s2.get('sharpe', float('nan')):5.2f}")
    return (f"{nom:26} {st['rend_ann']*100:6.2f}% {st['vol_ann']*100:5.1f}% "
            f"{st['sharpe']:6.2f} {st['t_stat']:6.2f} {st['dd_max']*100:6.1f}% | "
            + " ".join(per))


def _alphas_marche(args: tuple[str, pd.Series]) -> tuple[str, dict[str, pd.Series]]:
    sid, close = args
    reg = walk_forward_hmm(close)
    jm = walk_forward_jump(close)
    ts = D.alpha_tsmom(close)
    st_h = reg["state"]
    allow_h = np.where(np.sign(ts) * st_h.fillna(0) < 0, 0, np.sign(ts).fillna(0)).astype(int)
    allow_h[st_h.isna().to_numpy()] = 0
    variantes = {
        "1 TSMOM 12 mois": ts,
        "2 Tendance 1/3/12 mois": D.alpha_trend_mix(close),
        "3 HMM seul (forward)": D.alpha_regime(reg),
        "4 HMM Viterbi seul": reg["viterbi"],
        "5 TSMOM + filtre HMM": D.filtre_regime(ts, st_h),
        "6 TSMOM + filtre sauts": D.filtre_regime(ts, jm["state"]),
        "7 Fibonacci seul": fibo_positions(close, None),
        "8 Fibo + TSMOM + HMM": fibo_positions(close, allow_h),
    }
    return sid, {nom: D.rendements_strategie(close, sig, FRED[sid])
                 for nom, sig in variantes.items()}


def mode_daily(dossier: str, workers: int = 4) -> None:
    from concurrent.futures import ProcessPoolExecutor

    series = D.charger_fred(dossier, list(FRED), depuis="1985")
    alphas: dict[str, dict[str, pd.Series]] = {}
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for sid, res in ex.map(_alphas_marche, series.items()):
            for nom, r in res.items():
                alphas.setdefault(nom, {})[sid] = r
            print(f"  {sid}: {len(series[sid])} jours ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n{'variante':26} {'rend/an':>7} {'vol':>6} {'Sharpe':>6} {'t':>6} {'DD max':>7} | "
          + " ".join(f"{p:>9}" for p in PERIODES))
    for nom, par in alphas.items():
        print(_ligne(nom, D.portefeuille(par)))
        for sid, r in par.items():
            st = D.stats(r)
            if "sharpe" in st:
                print(f"      {sid:14} Sharpe {st['sharpe']:5.2f}", flush=True)


COUTS_H1 = {"frxXAUUSD": 0.16 / 500, "frxEURUSD": 0.10 / 500, "R_75": 0.29 / 1000,
            "frxGBPUSD": 0.10 / 500, "frxUSDJPY": 0.14 / 500, "frxAUDUSD": 0.20 / 500,
            "frxUSDCAD": 0.11 / 500, "frxUSDCHF": 0.15 / 500, "frxEURJPY": 0.13 / 500,
            "frxGBPJPY": 0.12 / 500}


def mode_h1(paires: list[tuple[str, str]], sims: int = 100) -> None:
    rules = ExitRules(max_hold=120)
    tous: dict[str, list[float]] = {}
    for path, sym in paires:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        cost = COUTS_H1[sym]
        reg = walk_forward_hmm(df["close"], train_len=1500, refit_every=250)
        tend = np.sign(df["close"] / df["close"].shift(240) - 1).fillna(0)
        st_h = reg["state"]
        hmm_dir = st_h.fillna(0).astype(int).to_numpy()
        avec_tend = np.where(np.sign(tend) * st_h.fillna(0) < 0, 0, tend).astype(int)
        avec_tend[st_h.isna().to_numpy()] = 0
        debut = int(st_h.notna().to_numpy().argmax())
        variantes = {
            "Fibo seul": None,
            "Fibo + HMM": np.where(hmm_dir == 0, 0, hmm_dir),
            "Fibo + tendance + HMM": avec_tend,
        }
        print(f"\n{sym} ({len(df)} bougies, test à partir de {df.index[debut]:%Y-%m-%d})")
        for nom, allow in variantes.items():
            if allow is not None:
                allow = allow.copy()
                allow[:debut] = 0
            sigs = fibo_signals(df, allow=allow)
            sigs = {i: s for i, s in sigs.items() if i >= debut}
            res = run(df, sigs, rules, cost, debut, len(df))
            st = res.stats()
            tous.setdefault(nom, []).extend(res.rets.tolist())
            if st["n"] == 0:
                print(f"  {nom:24} aucun trade")
                continue
            base = random_baseline(df, sigs, rules, cost, debut, len(df), n_sims=sims)
            p = float((base >= st["moy_bps"] / 1e4).mean()) if base.size else float("nan")
            print(f"  {nom:24} {st['n']:4} trades  réussite {st['wr']:.1%}  "
                  f"{st['moy_bps']:+6.1f} bp/trade  PF {st['pf']:.2f}  | hasard "
                  f"{base.mean()*1e4:+6.1f} bp  p={p:.2f}")
    print("\nTOUS MARCHÉS CONFONDUS")
    for nom, r in tous.items():
        r = np.array(r)
        if r.size:
            t = r.mean() / (r.std(ddof=1) / np.sqrt(r.size)) if r.size > 1 else float("nan")
            print(f"  {nom:24} {r.size:4} trades  réussite {(r > 0).mean():.1%}  "
                  f"{r.mean()*1e4:+6.1f} bp/trade  t={t:+.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "daily":
        mode_daily(sys.argv[2])
    else:
        args = sys.argv[2:]
        mode_h1(list(zip(args[::2], args[1::2])))
