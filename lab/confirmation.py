"""Essai « délai de confirmation » : le signal de tendance doit tenir N jours avant d'être suivi.

    python -m lab.confirmation data/fred data/couts_mt5.json

Recherche seulement : rien n'est branché au bot. Backtest de production (régime inclus,
coûts Deriv : demi-spread + surcoût de swap), seul le signal est remplacé. Idée testée :
« éviter les faux mouvements » en n'ouvrant (ou ne retournant) qu'après N clôtures d'accord.
Causal : le signal confirmé du jour t n'utilise que les signaux ≤ t.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from trend_bot.backtest import Options, backtest, proba_meilleur, sharpe_degonfle, stats
from trend_bot.data import MARCHE, charger_fred
from trend_bot.signals import per_market, trend_signal, weekdays_only

SERIES = ["DEXUSEU", "DEXUSUK", "DEXJPUS", "DEXUSAL", "DEXCAUS", "DEXSZUS", "DEXUSNZ",
          "DEXMXUS", "DEXNOUS", "DEXSDUS", "NASDAQCOM"]
DELAIS = (3, 5, 10)


def confirme(sig: pd.DataFrame, n: int) -> pd.DataFrame:
    """Garde la dernière valeur tenue n jours de suite ; sinon conserve la valeur confirmée
    précédente (données ≤ t uniquement)."""
    stable = pd.DataFrame(True, index=sig.index, columns=sig.columns)
    for k in range(1, n):
        stable &= sig.eq(sig.shift(k)) & sig.notna()
    return sig.where(stable).ffill()


def _run(args):
    nom, closes, sig, couts, surcout, regimes = args
    import trend_bot.backtest as bt
    bt.market_signal = lambda c, strat, _sig=sig: _sig[c.name].reindex(c.index)
    res = backtest(closes, couts, Options(), regimes=regimes, swaps=surcout)
    return nom, res.rets


def main(dossier: str, couts_mt5: str) -> None:
    closes = weekdays_only(pd.DataFrame(charger_fred(Path(dossier), SERIES, "1985")).sort_index())
    cm = json.loads(Path(couts_mt5).read_text())["marches"]
    couts = {s: cm[MARCHE[s]]["cout"] for s in SERIES}
    surcout = {s: cm[MARCHE[s]]["surcout_swap"] for s in SERIES}
    regimes = pd.read_pickle(Path("data") / "regimes_1985.pkl").reindex(closes.index)[SERIES]
    tend = per_market(trend_signal, closes)
    sigs = {"T  tendance (production)": tend}
    for n in DELAIS:
        sigs[f"C{n} confirmé {n} jours"] = confirme(tend, n)
    taches = [(nom, closes, sig, couts, surcout, regimes) for nom, sig in sigs.items()]
    with ProcessPoolExecutor(min(4, len(taches))) as ex:
        res = dict(ex.map(_run, taches))
    ref = res["T  tendance (production)"]
    print(f"Panier de production, coûts Deriv, {closes.index[0].year}-{closes.index[-1].year}")
    print(f"{'variante':28} {'Sharpe':>7} {'CAGR':>7} {'DD':>5} | {'1996-26':>7} | {'2008-26':>7} | {'P>T':>5} | rotation")
    essais = []
    for nom, r in res.items():
        s, s96, s08 = stats(r), stats(r["1996":]), stats(r["2008":])
        essais.append(r.mean() / r.std())
        p = proba_meilleur(r, ref) if nom != "T  tendance (production)" else float("nan")
        print(f"{nom:28} {s['sharpe']:+7.2f} {s['cagr']:+7.1%} {s['dd_max']:5.0%} | "
              f"{s96['sharpe']:+7.2f} | {s08['sharpe']:+7.2f} | {p:5.2f}")
    meilleur = max((n for n in res if not n.startswith("T ")), key=lambda n: stats(res[n])["sharpe"])
    print(f"Sharpe dégonflé du meilleur essai ({meilleur.split()[0]}, {len(DELAIS)} essais) : "
          f"{sharpe_degonfle(res[meilleur], essais[1:]):.2f} (seuil 0,95)")
    pd.to_pickle(res, Path("data") / "confirmation_rets.pkl")


if __name__ == "__main__":
    main(*sys.argv[1:3])
