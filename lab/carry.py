"""Signal de portage (carry) : différentiel de taux 3 mois, seul et combiné à la tendance.

    python -m lab.carry data/fred data/couts_mt5.json

Recherche seulement : rien ici n'est branché au bot. Le backtest est celui de production
(`trend_bot.backtest.backtest`, régime inclus) ; seul le signal est remplacé.

Nouveauté par rapport aux rapports précédents : le P&L du portage (différentiel de taux
porté par la position) est AJOUTÉ, ce que le backtest sur prix au comptant ignorait.
  - scénario « Deriv »   : spread + portage − surcoût du courtier sur le swap ;
  - scénario « futures » : spread + portage, aucun surcoût (le portage est dans le prix).
Taux : FRED IR3TIB01xxM156N (interbancaire 3 mois, mensuel), décalés d'un mois (publication).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from trend_bot.backtest import Options, backtest, proba_meilleur, sharpe_degonfle, stats
from trend_bot.data import MARCHE, charger_fred
from trend_bot.signals import ANN, ex_ante_vol, per_market, trend_signal, weekdays_only

TAUX = {"USD": "IR3TIB01USM156N", "EUR": "IR3TIB01EZM156N", "JPY": "IR3TIB01JPM156N",
        "GBP": "IR3TIB01GBM156N", "AUD": "IR3TIB01AUM156N", "CAD": "IR3TIB01CAM156N",
        "CHF": "IR3TIB01CHM156N", "NZD": "IR3TIB01NZM156N", "MXN": "IR3TIB01MXM156N",
        "NOK": "IR3TIB01NOM156N", "SEK": "IR3TIB01SEM156N"}
# série FRED → (devise achetée quand on est long, devise vendue)
PAIRES = {"DEXUSEU": ("EUR", "USD"), "DEXUSUK": ("GBP", "USD"), "DEXUSAL": ("AUD", "USD"),
          "DEXUSNZ": ("NZD", "USD"), "DEXJPUS": ("USD", "JPY"), "DEXCAUS": ("USD", "CAD"),
          "DEXSZUS": ("USD", "CHF"), "DEXMXUS": ("USD", "MXN"), "DEXNOUS": ("USD", "NOK"),
          "DEXSDUS": ("USD", "SEK")}


def taux_journaliers(dossier: Path, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Taux mensuels → quotidiens, en fraction annuelle, connus avec un mois de retard."""
    out = {}
    for dev, sid in TAUX.items():
        s = pd.read_csv(dossier / f"{sid}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce").dropna() / 100
        s.index = s.index + pd.offsets.MonthBegin(1)        # la valeur du mois m est connue au 1er du mois m+1
        out[dev] = s.reindex(index.union(s.index)).ffill().reindex(index)
    return pd.DataFrame(out)


def portage(taux: pd.DataFrame, series: list[str]) -> pd.DataFrame:
    """Différentiel de taux annuel porté par une position LONGUE sur chaque série (positif = on
    encaisse)."""
    return pd.DataFrame({s: taux[PAIRES[s][0]] - taux[PAIRES[s][1]] for s in series})


def signaux(closes: pd.DataFrame, carry: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Les variantes testées, toutes dans [-1, 1], calculées avec des données ≤ t."""
    tend = per_market(trend_signal, closes)
    vol = per_market(ex_ante_vol, closes)
    c_signe = np.sign(carry).where(carry.notna())
    c_vol = (carry / vol).clip(-1, 1)                      # portage rapporté au risque (Carver)
    return {
        "T  tendance (production)": tend,
        "C1 portage : signe": c_signe,
        "C2 portage / volatilité": c_vol,
        "T+C1 moyenne tendance et signe du portage": (tend + c_signe) / 2,
        "T+C2 moyenne tendance et portage/vol": (tend + c_vol) / 2,
    }


def _run(args):
    """Un backtest : signal remplacé par une série précalculée, P&L du portage ajouté."""
    nom, scenario, closes, sig, carry, couts, surcout, regimes = args
    import trend_bot.backtest as bt
    bt.market_signal = lambda c, strat, _sig=sig: _sig[c.name].reindex(c.index)
    res = backtest(closes, couts, Options(), regimes=regimes,
                   swaps=surcout if scenario == "Deriv" else None)
    w = res.targets.reindex(res.rets.index).shift(1)       # poids décidés en t, portés t → t+1
    carry_pnl = (w * carry.reindex(res.rets.index).shift(1).fillna(0.0) / ANN).sum(axis=1)
    return nom, scenario, res.rets + carry_pnl, res.rets


def main(dossier: str, couts_mt5: str, depuis: str = "1990") -> None:
    dossier = Path(dossier)
    series = list(PAIRES)
    closes = weekdays_only(pd.DataFrame(charger_fred(dossier, series, depuis)).sort_index())
    taux = taux_journaliers(dossier, closes.index)
    carry = portage(taux, series)
    debut = carry.dropna().index[0]
    print(f"Portage disponible pour les 10 paires à partir du {debut.date()} (JPY : FRED depuis 2002).")
    cm = json.loads(Path(couts_mt5).read_text())["marches"]
    couts = {s: cm[MARCHE[s]]["cout"] for s in series}
    surcout = {s: cm[MARCHE[s]]["surcout_swap"] for s in series}
    regimes = pd.read_pickle(Path("data") / "regimes_1985.pkl").reindex(closes.index)[series]
    sigs = signaux(closes, carry)
    taches = [(nom, sc, closes, sig, carry, couts, surcout, regimes)
              for nom, sig in sigs.items() for sc in ("Deriv", "futures")]
    res = {}
    with ProcessPoolExecutor(min(7, len(taches))) as ex:
        for nom, sc, r, r_spot in ex.map(_run, taches):
            res[(nom, sc)] = (r, r_spot)
    fin = closes.index[-1]
    for sc in ("Deriv", "futures"):
        print(f"\n=== Scénario {sc} : {debut.year}-{fin.year}, portage encaissé"
              + (" moins surcoût de swap" if sc == "Deriv" else ", sans surcoût") + " ===")
        print(f"{'variante':46} {'Sharpe':>7} {'CAGR':>7} {'DD':>5} | {'2008-26':>7} | {'P>T':>5} | spot seul")
        ref = res[("T  tendance (production)", sc)][0][debut:]
        essais = []
        for nom in sigs:
            r, r_spot = res[(nom, sc)]
            r, r_spot = r[debut:], r_spot[debut:]
            s, s08, ss = stats(r), stats(r["2008":]), stats(r_spot)
            essais.append(r.mean() / r.std())
            p = proba_meilleur(r, ref) if nom != "T  tendance (production)" else float("nan")
            print(f"{nom:46} {s['sharpe']:+7.2f} {s['cagr']:+7.1%} {s['dd_max']:5.0%} | "
                  f"{s08['sharpe']:+7.2f} | {p:5.2f} | {ss['sharpe']:+.2f}")
        meilleur = max((n for n in sigs if n != "T  tendance (production)"),
                       key=lambda n: stats(res[(n, sc)][0][debut:])["sharpe"])
        print(f"Sharpe dégonflé du meilleur essai ({meilleur.split()[0]}, {len(sigs) - 1} essais) : "
              f"{sharpe_degonfle(res[(meilleur, sc)][0][debut:], essais[1:]):.2f} (seuil 0,95)")
    pd.to_pickle({k: v[0] for k, v in res.items()}, Path("data") / "carry_rets.pkl")


if __name__ == "__main__":
    main(*sys.argv[1:3])
