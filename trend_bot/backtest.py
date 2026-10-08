"""Rejoue la logique EXACTE du bot jour par jour : mêmes fonctions que le runner.

Jour t (clôture) : signaux → filtre de régime → poids cibles (volatilité cible du
portefeuille) → coupe-circuit → zone neutre. Le rendement est celui de t à t+1 ;
les frais sont payés sur chaque variation de poids.

Usage : python -m trend_bot.backtest data/fred
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trend_bot.portfolio import BUFFER, needs_trade, target_weights
from trend_bot.risk import KillSwitch, apply_regime, regime_series
from trend_bot.signals import ANN, daily_returns, ex_ante_vol, per_market, trend_signal

WEIGHT_STEP = 0.002        # plus petit ajustement de poids simulé (≈ un pas de volume)


@dataclass
class Options:
    regime: bool = True
    killswitch: bool = True
    buffer: float = BUFFER
    refit_every: int = 21
    soft_dd: float = 0.10
    hard_dd: float = 0.20
    pause_jours: int = 21      # backtest : durée à plat avant la remise à zéro manuelle
    rebalance_every: int = 1   # jours entre deux rééquilibrages (le risque est vérifié chaque jour)


@dataclass
class Resultat:
    equity: pd.Series
    rets: pd.Series
    gross: pd.Series
    turnover: pd.Series
    trips: list = field(default_factory=list)


def backtest(closes: pd.DataFrame, costs: dict[str, float], opt: Options = Options(),
             regimes: pd.DataFrame | None = None) -> Resultat:
    closes = closes.sort_index()
    rets = daily_returns(closes)
    sig_all = per_market(trend_signal, closes)
    vol_all = per_market(ex_ante_vol, closes)
    if opt.regime and regimes is None:
        regimes = pd.DataFrame({c: regime_series(closes[c].dropna(), refit_every=opt.refit_every)
                                for c in closes}).reindex(closes.index).ffill(limit=10)
    cost = pd.Series(costs).reindex(closes.columns).fillna(5e-4)
    ks = KillSwitch(soft=opt.soft_dd, hard=opt.hard_dd)
    w = pd.Series(0.0, index=closes.columns)
    eq, out_r, out_g, out_t, trips, pause = 1.0, [], [], [], [], 0
    dates = closes.index
    start = int(sig_all.notna().any(axis=1).to_numpy().argmax())
    for t in range(start, len(dates) - 1):
        sig = sig_all.iloc[t].copy()
        if opt.regime:
            st = regimes.iloc[t]
            sig = pd.Series({c: apply_regime(sig[c], None if np.isnan(st[c]) else int(st[c]))
                             for c in sig.index})
        cible = target_weights(sig, vol_all.iloc[t], rets.iloc[max(0, t - 300):t + 1])
        if opt.killswitch:
            if pause > 0:                      # à plat après un déclenchement
                pause -= 1
                if pause == 0:
                    ks.reset(eq)               # en réel : remise à zéro manuelle
            mult = 0.0 if pause > 0 else ks.update(eq)
            if ks.tripped and pause == 0:
                trips.append((dates[t], eq))
                pause = opt.pause_jours
            cible *= mult
        neuf = w.copy()
        jour_de_rebal = (t - start) % opt.rebalance_every == 0
        for c in closes.columns:
            coupe = cible[c] == 0 and w[c] != 0 and opt.killswitch and pause > 0
            if (jour_de_rebal or coupe) and needs_trade(w[c], cible[c], WEIGHT_STEP, opt.buffer):
                neuf[c] = cible[c]
        frais = float((cost * (neuf - w).abs()).sum())
        pnl = float((neuf * rets.iloc[t + 1].fillna(0.0)).sum()) - frais
        eq *= 1 + pnl
        out_r.append(pnl)
        out_g.append(float(neuf.abs().sum()))
        out_t.append(float((neuf - w).abs().sum()))
        w = neuf
    idx = dates[start + 1:]
    r = pd.Series(out_r, index=idx)
    return Resultat((1 + r).cumprod(), r, pd.Series(out_g, index=idx),
                    pd.Series(out_t, index=idx), trips)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {}
    eq = (1 + r).cumprod()
    sharpe = r.mean() / r.std() * np.sqrt(ANN)
    ans = len(r) / ANN
    return {
        "cagr": eq.iloc[-1] ** (1 / ans) - 1,
        "vol": r.std() * np.sqrt(ANN),
        "sharpe": sharpe,
        "t": sharpe * np.sqrt(ans),
        "dd_max": float((1 - eq / eq.cummax()).max()),
    }


def _main(dossier: str) -> None:
    from lab.daily import charger_fred
    from lab.run_systeme import FRED

    series = charger_fred(dossier, list(FRED), depuis="1985")
    closes = pd.DataFrame(series)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        regs = dict(zip(closes, ex.map(regime_series, [closes[c].dropna() for c in closes])))
    regimes = pd.DataFrame(regs).reindex(closes.index).ffill(limit=10)
    print(f"régimes calculés en {time.time() - t0:.0f}s")
    variantes = {
        "A tendance seule": Options(regime=False, killswitch=False),
        "B + filtre de régime": Options(killswitch=False),
        "C + régime + coupe-circuit (BOT)": Options(),
        "D bot sans zone neutre": Options(buffer=0.0),
    }
    periodes = [("1990", "2007"), ("2008", "2016"), ("2017", "2026")]
    print(f"\n{'variante':34} {'CAGR':>6} {'vol':>5} {'Sharpe':>6} {'t':>5} {'DD max':>6} "
          f"{'rotation/an':>11} {'levier moy':>10} | " + " ".join(f"{a}-{b[2:]}" for a, b in periodes))
    for nom, opt in variantes.items():
        res = backtest(closes, FRED, opt, regimes=regimes)
        s = stats(res.rets)
        per = " ".join(f"{stats(res.rets[a:b]).get('sharpe', np.nan):7.2f}" for a, b in periodes)
        print(f"{nom:34} {s['cagr']*100:5.1f}% {s['vol']*100:4.1f}% {s['sharpe']:6.2f} "
              f"{s['t']:5.2f} {s['dd_max']*100:5.1f}% {res.turnover.mean()*ANN:11.1f} "
              f"{res.gross.mean():10.2f} | {per}"
              + (f"  [coupe-circuit : {len(res.trips)}×]" if res.trips else ""))


if __name__ == "__main__":
    _main(sys.argv[1] if len(sys.argv) > 1 else "data/fred")
