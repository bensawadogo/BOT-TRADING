"""Rejoue la logique EXACTE du bot jour par jour : mêmes fonctions que le runner.

Jour t (clôture) : signaux → filtre de régime → poids cibles (volatilité cible du
portefeuille) → coupe-circuit → zone neutre. Le rendement est celui de t à t+1 ;
les frais sont payés sur chaque variation de poids.

Usage : python -m trend_bot.backtest data/fred [data/couts_mt5.json]
  Le second argument (scripts/couts_mt5.py tendance) ajoute les variantes aux coûts
  réels du courtier : demi-spread relevé et surcoût des swaps.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from trend_bot.portfolio import BUFFER, needs_trade, target_weights
from trend_bot.risk import KillSwitch, apply_regime, regime_series
from trend_bot.signals import (ANN, Strategie, daily_returns, ex_ante_vol, market_signal,
                               per_market, weekdays_only)

WEIGHT_STEP = 0.002        # plus petit ajustement de poids simulé (≈ un pas de volume)


@dataclass
class Options:
    regime: bool = True
    killswitch: bool = True
    buffer: float = BUFFER
    soft_dd: float = 0.10
    hard_dd: float = 0.20
    pause_jours: int = 21      # backtest : durée à plat avant la remise à zéro manuelle
    rebalance_every: int = 1   # jours entre deux rééquilibrages (le risque est vérifié chaque jour)
    strat: Strategie = Strategie()   # direction / confiance / régime : partagé avec le live
    fibo_gate: bool = False    # CONTRÔLE (essai E4, backtest seulement) : n'ouvrir qu'en zone Fibonacci


@dataclass
class Resultat:
    equity: pd.Series
    rets: pd.Series
    gross: pd.Series
    turnover: pd.Series
    targets: pd.DataFrame       # poids cibles du jour (après coupe-circuit), = ceux du live
    trips: list = field(default_factory=list)


def backtest(closes: pd.DataFrame, costs: dict[str, float], opt: Options = Options(),
             regimes: pd.DataFrame | None = None,
             swaps: dict[str, float] | None = None,
             admin: dict[str, tuple[float, float]] | None = None,
             grace: int = 15) -> Resultat:
    """`costs` : coût par unité de rotation ; `swaps` : coût annuel des positions tenues,
    en fraction du notionnel (surcoût du courtier sur le différentiel de taux) ;
    `admin` : (long, court) annuel des frais d'administration du compte sans swap, dus
    une fois la position tenue plus de `grace` jours calendaires dans le même sens."""
    closes = weekdays_only(closes.sort_index())
    rets = daily_returns(closes)
    sig_all = per_market(lambda c: market_signal(c, opt.strat), closes)
    vol_all = per_market(ex_ante_vol, closes)
    if opt.fibo_gate:
        from lab.fibo import zone_series
        zones = {c: zone_series(closes[c].dropna(),
                                (closes[c].dropna() * ex_ante_vol(closes[c].dropna())
                                 / np.sqrt(ANN)).to_numpy()) for c in closes}
        gate_l = pd.DataFrame({c: z["long"] for c, z in zones.items()}).reindex(
            closes.index).ffill(limit=10).fillna(False).astype(bool)
        gate_s = pd.DataFrame({c: z["short"] for c, z in zones.items()}).reindex(
            closes.index).ffill(limit=10).fillna(False).astype(bool)
    if opt.regime and regimes is None:
        regimes = pd.DataFrame({c: regime_series(closes[c].dropna()) for c in closes}
                               ).reindex(closes.index).ffill(limit=10)
    cost = pd.Series(costs).reindex(closes.columns).fillna(5e-4)
    swap = pd.Series(swaps or {}).reindex(closes.columns).fillna(0.0) / ANN
    ks = KillSwitch(soft=opt.soft_dd, hard=opt.hard_dd)
    adm_l = pd.Series({c: a[0] for c, a in (admin or {}).items()}).reindex(closes.columns).fillna(0.0)
    adm_c = pd.Series({c: a[1] for c, a in (admin or {}).items()}).reindex(closes.columns).fillna(0.0)
    age = pd.Series(0, index=closes.columns)           # jours de bourse dans le même sens
    seuil = int(np.ceil(grace * ANN / 365))
    w = pd.Series(0.0, index=closes.columns)
    eq, out_r, out_g, out_t, out_c, trips, pause = 1.0, [], [], [], [], [], 0
    dates = closes.index
    start = int(sig_all.notna().any(axis=1).to_numpy().argmax())
    for t in range(start, len(dates) - 1):
        sig = sig_all.iloc[t].copy()
        if opt.regime:
            st = regimes.iloc[t]
            sig = pd.Series({c: apply_regime(sig[c], None if np.isnan(st[c]) else int(st[c]),
                                             opt.strat.regime_facteur) for c in sig.index})
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
        out_c.append(cible)
        neuf = w.copy()
        jour_de_rebal = (t - start) % opt.rebalance_every == 0
        for c in closes.columns:
            coupe = cible[c] == 0 and w[c] != 0 and opt.killswitch and pause > 0
            if (jour_de_rebal or coupe) and needs_trade(w[c], cible[c], WEIGHT_STEP, opt.buffer):
                if (opt.fibo_gate and cible[c] != 0 and np.sign(cible[c]) != np.sign(w[c])
                        and not (gate_l if cible[c] > 0 else gate_s).iloc[t][c]):
                    neuf[c] = 0.0              # on attend le retracement : à plat d'ici là
                    continue
                neuf[c] = cible[c]
        age = age.where(np.sign(neuf) == np.sign(w), 0) + (neuf != 0)
        taux = adm_l.where(neuf > 0, adm_c) * (age > seuil)
        frais = float((cost * (neuf - w).abs()).sum() + (swap * neuf.abs()).sum()
                      + (taux * neuf.abs()).sum() / ANN)
        pnl = float((neuf * rets.iloc[t + 1].fillna(0.0)).sum()) - frais
        eq *= 1 + pnl
        out_r.append(pnl)
        out_g.append(float(neuf.abs().sum()))
        out_t.append(float((neuf - w).abs().sum()))
        w = neuf
    idx = dates[start + 1:]
    r = pd.Series(out_r, index=idx)
    return Resultat((1 + r).cumprod(), r, pd.Series(out_g, index=idx),
                    pd.Series(out_t, index=idx),
                    pd.DataFrame(out_c, index=dates[start:len(dates) - 1]), trips)


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


def sharpe_degonfle(r: pd.Series, sharpes_essais: list[float]) -> float:
    """Probabilité que le vrai Sharpe soit > 0 après correction du nombre d'essais et de la
    forme des rendements (Bailey et López de Prado, 2014). `sharpes_essais` : Sharpe
    QUOTIDIEN de chacun des essais tentés sur la même période. > 0,95 = significatif."""
    from scipy.stats import kurtosis, norm, skew

    r = r.dropna()
    n, sr = len(r), r.mean() / r.std()
    k = len(sharpes_essais)
    euler = 0.5772156649
    sr0 = np.std(sharpes_essais, ddof=1) * ((1 - euler) * norm.ppf(1 - 1 / k)
                                            + euler * norm.ppf(1 - 1 / (k * np.e)))
    denom = np.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))


def proba_meilleur(r: pd.Series, ref: pd.Series, tirages: int = 4000, bloc: int = 21,
                   seed: int = 0) -> float:
    """Part des rééchantillonnages (blocs de `bloc` jours, mêmes dates pour les deux
    séries) où le Sharpe de `r` dépasse celui de `ref` : proche de 0,5 = indiscernable."""
    r, ref = r.align(ref, join="inner")
    a, b, n = r.to_numpy(), ref.to_numpy(), len(r)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / bloc))
    debuts = rng.integers(0, n, (tirages, nb))
    idx = ((debuts[:, :, None] + np.arange(bloc)) % n).reshape(tirages, -1)[:, :n]
    sa, sb = a[idx], b[idx]
    return float(((sa.mean(1) / sa.std(1)) > (sb.mean(1) / sb.std(1))).mean())


def _ligne(nom: str, res: Resultat, periodes) -> str:
    s = stats(res.rets)
    per = " ".join(f"{stats(res.rets[a:b]).get('sharpe', np.nan):7.2f}" for a, b in periodes)
    return (f"{nom:44} {s['cagr']*100:5.1f}% {s['vol']*100:4.1f}% {s['sharpe']:6.2f} "
            f"{s['t']:5.2f} {s['dd_max']*100:5.1f}% {res.turnover.mean()*ANN:11.1f} "
            f"{res.gross.mean():10.2f} | {per}"
            + (f"  [coupe-circuit : {len(res.trips)}×]" if res.trips else ""))


ESSAIS = {   # fixés à l'avance (docs/reports/COMBINER_LES_ALGOS_2026-11-03.md, § 4) : 5 essais
    "E0 bot d'origine (signes 1/3/12 mois)": Options(),
    "E1 tendance continue 3 vitesses": Options(strat=Strategie("ewmac")),
    "E2 E1 + désaccord lent/rapide ×0,5": Options(strat=Strategie("ewmac", 0.5)),
    "E3 E2 + régime : moitié au lieu de couper": Options(strat=Strategie("ewmac", 0.5, 0.5)),
    "E4 E0 + Fibonacci (timing d'entrée)": Options(fibo_gate=True),
}


def _essais(closes: pd.DataFrame, regimes: pd.DataFrame, couts: dict, swaps: dict | None,
            titre: str) -> dict:
    """Passe les 5 essais dans un même scénario de coûts ; décision sur 2008-2026."""
    res = {nom: backtest(closes, couts, opt, regimes=regimes, swaps=swaps)
           for nom, opt in ESSAIS.items()}
    periodes = [("1990", "2007"), ("2008", "2026")]
    srs = [stats(r.rets["2008":])["sharpe"] / np.sqrt(ANN) for r in res.values()]
    ref = next(iter(res.values())).rets["2008":]
    print(f"\n{titre}\n{'essai':44} {'CAGR':>6} {'Sharpe':>6} {'DD max':>6} {'rot./an':>7} "
          f"| {'90-07':>6} {'08-26':>6} {'DSR':>5} {'P>E0':>5}")
    out = {}
    for nom, r in res.items():
        s_all, s_oos = stats(r.rets), stats(r.rets["2008":])
        dsr = sharpe_degonfle(r.rets["2008":], srs)
        pb = proba_meilleur(r.rets["2008":], ref)
        sh = [stats(r.rets[a:b]).get("sharpe", np.nan) for a, b in periodes]
        print(f"{nom:44} {s_all['cagr']*100:5.1f}% {s_all['sharpe']:6.2f} "
              f"{s_all['dd_max']*100:5.1f}% {r.turnover.mean()*ANN:7.1f} | "
              f"{sh[0]:6.2f} {sh[1]:6.2f} {dsr:5.2f} {pb:5.2f}")
        out[nom] = dict(sharpe=s_all["sharpe"], oos=s_oos["sharpe"], dsr=dsr, pb=pb)
    return out


def _main_essais(dossier: str, couts_mt5: str) -> None:
    import json

    from trend_bot.data import FRED, MARCHE, charger_fred

    closes = pd.DataFrame(charger_fred(dossier, list(FRED), depuis="1985"))
    cache = Path(dossier).parent / "regimes_1985.pkl"
    regimes = pd.read_pickle(cache) if cache.exists() else None
    if regimes is None or list(regimes.columns) != list(closes.columns)             or regimes.index[-1] != closes.index[-1]:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(4) as ex:
            regs = dict(zip(closes, ex.map(regime_series, [closes[c].dropna() for c in closes])))
        regimes = pd.DataFrame(regs).reindex(closes.index).ffill(limit=10)
        regimes.to_pickle(cache)
    releve = json.loads(Path(couts_mt5).read_text())["marches"]
    spread = {sid: releve[m]["cout"] for sid, m in MARCHE.items() if m in releve}
    swaps = {sid: releve[m]["surcout_swap"] for sid, m in MARCHE.items() if m in releve}
    cols = [sid for sid in closes if MARCHE[sid] not in ("USOIL", "UKOIL", "NGAS")]
    _essais(closes, regimes, FRED, None, "A. 14 marchés, frais d'origine du backtest")
    _essais(closes[cols], regimes[cols], spread, swaps,
            "B. 11 marchés du bot, spreads réels + swaps (compte Deriv standard)")
    _essais(closes[cols], regimes[cols], spread, None,
            "C. 11 marchés du bot, spreads réels, sans swap (compte sans swap)")


def _main(dossier: str, couts_mt5: str | None = None) -> None:
    import json

    from trend_bot.data import FRED, MARCHE, charger_fred

    series = charger_fred(dossier, list(FRED), depuis="1985")
    closes = pd.DataFrame(series)
    t0 = time.time()
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(4) as ex:
        regs = dict(zip(closes, ex.map(regime_series, [closes[c].dropna() for c in closes])))
    regimes = pd.DataFrame(regs).reindex(closes.index).ffill(limit=10)
    print(f"régimes calculés en {time.time() - t0:.0f}s")
    periodes = [("1990", "2007"), ("2008", "2016"), ("2017", "2026")]
    print(f"\n{'variante':44} {'CAGR':>6} {'vol':>5} {'Sharpe':>6} {'t':>5} {'DD max':>6} "
          f"{'rotation/an':>11} {'levier moy':>10} | " + " ".join(f"{a}-{b[2:]}" for a, b in periodes))
    variantes = {
        "A tendance seule": Options(regime=False, killswitch=False),
        "B + filtre de régime": Options(killswitch=False),
        "C + régime + coupe-circuit (BOT)": Options(),
        "D bot sans zone neutre": Options(buffer=0.0),
    }
    for nom, opt in variantes.items():
        print(_ligne(nom, backtest(closes, FRED, opt, regimes=regimes), periodes))
    if not couts_mt5:
        return
    releve = json.loads(Path(couts_mt5).read_text())
    reel = releve["marches"]
    spread = {sid: reel[m]["cout"] for sid, m in MARCHE.items() if m in reel}
    swaps = {sid: reel[m]["surcout_swap"] for sid, m in MARCHE.items() if m in reel}
    sans_energie = [sid for sid in closes if MARCHE[sid] not in ("USOIL", "UKOIL", "NGAS")]
    print(f"\nCoûts réels du courtier (relevé {releve['releve_utc']}) :")
    for nom, cols, sw in [
        ("C 14 marchés, spreads réels, sans swap", list(closes), None),
        ("C 14 marchés, spreads réels + swaps", list(closes), swaps),
        ("C sans énergies, spreads réels, sans swap", sans_energie, None),
        ("C sans énergies, spreads réels + swaps", sans_energie, swaps),
    ]:
        res = backtest(closes[cols], spread, Options(), regimes=regimes[cols], swaps=sw)
        print(_ligne(nom, res, periodes))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--essais"]
    if "--essais" in sys.argv:
        _main_essais(*(args[:2] or ["data/fred", "data/couts_mt5.json"]))
    else:
        _main(*(args[:2] or ["data/fred"]))
