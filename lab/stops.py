"""Essai « stops » : un stop par position améliore-t-il le bot de tendance ?

    python -m lab.stops data/fred data/couts_mt5.json

Recherche seulement : rien n'est branché au bot. Backtest de production (régime, coupe-circuit,
coûts Deriv : demi-spread + surcoût de swap), seule la fonction des poids cibles est enveloppée
par une couche de stops, comme le ferait un ordre de stop placé chez le courtier : les autres
poids ne sont pas redimensionnés quand un marché est coupé.

Limites assumées : FRED ne donne que des clôtures journalières. Le stop est donc testé à la
clôture (pas de plus haut ni de plus bas intrajournalier), sans glissement ni gap.
Causal : l'état du jour t n'utilise que les rendements et la volatilité <= t.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from trend_bot.backtest import Options, backtest, proba_meilleur, sharpe_degonfle, stats
from trend_bot.data import MARCHE, charger_fred
from trend_bot.signals import ANN, daily_returns, weekdays_only

SERIES = ["DEXUSEU", "DEXUSUK", "DEXJPUS", "DEXUSAL", "DEXCAUS", "DEXSZUS", "DEXUSNZ",
          "DEXMXUS", "DEXNOUS", "DEXSDUS", "NASDAQCOM"]
MOIS = 21           # jours de bourse : le stop se mesure en écarts-types mensuels
DELAI = 21          # jours à plat avant de pouvoir rentrer (variante « délai »)


@dataclass(frozen=True)
class Variante:
    nom: str
    mode: str       # 'suiveur' : depuis le meilleur gain de la position ; 'entree' : depuis l'entrée
    mult: float     # distance du stop, en écarts-types mensuels de la volatilité ex-ante
    retour: str     # 'delai' : retour après DELAI jours si le signal n'a pas changé ;
    #                 'signal' : on attend que le sens du signal change


# S1 à S5 fixées avant de voir le moindre résultat. S6 et S7 ajoutées après le premier passage pour
# mesurer la sensibilité à la distance : les 7 essais sont comptés dans le Sharpe dégonflé.
VARIANTES = (
    Variante("S1 suiveur serré (1,5 σ mensuel), retour après 21 j", "suiveur", 1.5, "delai"),
    Variante("S2 suiveur large (3 σ mensuel), retour après 21 j", "suiveur", 3.0, "delai"),
    Variante("S3 catastrophe depuis l'entrée (5 σ mensuel)", "entree", 5.0, "delai"),
    Variante("S4 suiveur serré, attend un nouveau signal", "suiveur", 1.5, "signal"),
    Variante("S5 suiveur large, attend un nouveau signal", "suiveur", 3.0, "signal"),
    Variante("S6 suiveur très serré (1 σ mensuel), retour après 21 j", "suiveur", 1.0, "delai"),
    Variante("S7 suiveur moyen (2 σ mensuel), retour après 21 j", "suiveur", 2.0, "delai"),
)


class _Etat:
    __slots__ = ("cote", "cum", "best", "bloque", "cote_stop", "jours")

    def __init__(self):
        self.cote, self.cum, self.best = 0, 0.0, 0.0
        self.bloque, self.cote_stop, self.jours = False, 0, 0


class Superposition:
    """Enveloppe de `target_weights` : mêmes poids, plus des stops par position."""

    def __init__(self, base, variante: Variante):
        self.base, self.v = base, variante
        self.etats: dict[str, _Etat] = {}
        self.declenchements = 0
        self.appels = 0
        self.journal: list[tuple[int, str, int]] = []     # (numéro d'appel, marché, sens coupé)

    def __call__(self, signals, vols, returns):
        poids = self.base(signals, vols, returns).copy()
        r = returns.iloc[-1].fillna(0.0)
        self.appels += 1
        for c in poids.index:
            st = self.etats.setdefault(c, _Etat())
            if st.cote != 0:                      # position tenue pendant le jour écoulé
                st.cum += st.cote * float(r.get(c, 0.0))
                st.best = max(st.best, st.cum)
                dist = self.v.mult * vols[c] * np.sqrt(MOIS / ANN)
                ref = st.best if self.v.mode == "suiveur" else 0.0
                if np.isfinite(dist) and ref - st.cum > dist:
                    st.bloque, st.cote_stop, st.jours = True, st.cote, 0
                    self.journal.append((self.appels, c, st.cote))
                    st.cote = 0
                    self.declenchements += 1
            sens = int(np.sign(poids[c]))
            if st.bloque:
                st.jours += 1
                if sens != st.cote_stop or (self.v.retour == "delai" and st.jours >= DELAI):
                    st.bloque = False
            nouveau = 0 if st.bloque else sens
            if nouveau == 0:
                poids[c] = 0.0
            if nouveau != st.cote:                # nouvelle entrée ou fin de position
                st.cum, st.best = 0.0, 0.0
            st.cote = nouveau
        return poids


def restitution(targets: pd.DataFrame, rets: pd.DataFrame) -> dict:
    """Mesure du « les profits montent puis reviennent » sur des épisodes de position
    (un épisode = un marché dans le même sens, de l'ouverture à la fermeture)."""
    sens = np.sign(targets)
    r = rets.reindex(targets.index).shift(-1).fillna(0.0)      # le poids de t gagne le rendement t+1
    pics, finals = [], []
    for c in targets.columns:
        s, x = sens[c].to_numpy(), r[c].to_numpy()
        cum, pic, depart = 0.0, 0.0, None
        for i in range(len(s)):
            if depart is not None and s[i] != s[depart]:       # fin de l'épisode
                pics.append(pic)
                finals.append(cum)
                depart = None
            if s[i] == 0:
                continue
            if depart is None:
                depart, cum, pic = i, 0.0, 0.0
            cum += s[i] * x[i]
            pic = max(pic, cum)
        if depart is not None:
            pics.append(pic)
            finals.append(cum)
    pics, finals = np.array(pics), np.array(finals)
    gagnants = pics > 0.02                                       # a été en gain d'au moins 2 %
    return {
        "episodes": len(pics),
        "perdants": float((finals < 0).mean()),
        "rendus": float((pics[gagnants] - finals[gagnants]).sum() / pics[gagnants].sum())
        if gagnants.any() else float("nan"),
        "gagnants_finis_en_perte": float((finals[gagnants] < 0).mean())
        if gagnants.any() else float("nan"),
    }


def _run(args):
    nom, closes, variante, couts, surcout, regimes = args
    import trend_bot.backtest as bt
    from trend_bot.portfolio import target_weights as original   # jamais l'enveloppe d'une tâche précédente
    sup = None
    if variante is not None:
        sup = Superposition(original, variante)
        bt.target_weights = sup
    else:
        bt.target_weights = original
    res = backtest(closes, couts, Options(), regimes=regimes, swaps=surcout)
    rest = restitution(res.targets, daily_returns(closes))
    return nom, res.rets, rest, (sup.declenchements if sup else 0), res.trips


def main(dossier: str, couts_mt5: str) -> None:
    closes = weekdays_only(pd.DataFrame(charger_fred(Path(dossier), SERIES, "1985")).sort_index())
    cm = json.loads(Path(couts_mt5).read_text())["marches"]
    couts = {s: cm[MARCHE[s]]["cout"] for s in SERIES}
    surcout = {s: cm[MARCHE[s]]["surcout_swap"] for s in SERIES}
    regimes = pd.read_pickle(Path("data") / "regimes_1985.pkl").reindex(closes.index)[SERIES]
    taches = [("T  bot d'origine (sans stop)", closes, None, couts, surcout, regimes)]
    taches += [(v.nom, closes, v, couts, surcout, regimes) for v in VARIANTES]
    with ProcessPoolExecutor(4) as ex:
        sorties = list(ex.map(_run, taches))
    res = {nom: (r, rest, n, trips) for nom, r, rest, n, trips in sorties}
    ref = res["T  bot d'origine (sans stop)"][0]
    ans = len(ref) / ANN
    print(f"Panier de production, coûts Deriv, {closes.index[0].year}-{closes.index[-1].year}")
    print(f"{'variante':56} {'Sharpe':>6} {'CAGR':>6} {'DD':>4} | {'1996-26':>7} {'2008-26':>7} "
          f"| {'P>T':>4} | {'stops':>6} {'/an':>5} {'coupe-c.':>8}")
    essais = []
    for nom, (r, rest, n, trips) in res.items():
        s, s96, s08 = stats(r), stats(r["1996":]), stats(r["2008":])
        if not nom.startswith("T "):
            essais.append(r.mean() / r.std())
        p = proba_meilleur(r, ref) if not nom.startswith("T ") else float("nan")
        p_txt = "  = " if (n < 5 and not nom.startswith("T ")) else f"{p:4.2f}"   # quasi identique au bot
        print(f"{nom:56} {s['sharpe']:+6.2f} {s['cagr']:+6.1%} {s['dd_max']:4.0%} | "
              f"{s96['sharpe']:+7.2f} {s08['sharpe']:+7.2f} | {p_txt} | {n:6d} {n / ans:5.2f} {len(trips):8d}")
    print("\nProfits qui reviennent (épisodes de position, gain maximal d'au moins 2 %)")
    print(f"{'variante':56} {'épisodes':>8} {'finis en perte':>14} {'du pic rendu':>12} {'gagnants → perte':>16}")
    for nom, (r, rest, n, trips) in res.items():
        print(f"{nom:56} {rest['episodes']:8d} {rest['perdants']:14.0%} {rest['rendus']:12.0%} "
              f"{rest['gagnants_finis_en_perte']:16.0%}")
    meilleur = max((n for n in res if not n.startswith("T ")), key=lambda n: stats(res[n][0])["sharpe"])
    print(f"\nSharpe dégonflé du meilleur essai ({meilleur.split()[0]}, {len(VARIANTES)} essais) : "
          f"{sharpe_degonfle(res[meilleur][0], essais):.2f} (seuil 0,95)")
    pd.to_pickle({n: v[0] for n, v in res.items()}, Path("data") / "stops_rets.pkl")


if __name__ == "__main__":
    main(*sys.argv[1:3])
