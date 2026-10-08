"""
Portefeuille JOURNALIER multi-marchés, construit comme chez les gérants de tendance
(Moskowitz-Ooi-Pedersen 2012, Hurst-Ooi-Pedersen « A Century of Evidence on
Trend-Following ») et découpé comme le framework LEAN de QuantConnect :

  Alpha (signal -1..+1 par marché)  →  Construction (volatilité cible par marché,
  poids égal entre marchés)  →  Risque (levier plafonné)  →  Exécution (position
  décidée à la clôture t, rendement de t à t+1, coût sur chaque variation).

Données : clôtures journalières FRED (forex depuis 1971, pétrole, Nasdaq, gaz).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ANN = 252
VOL_CIBLE = 0.10          # 10 % annualisé par marché, comme AQR
LEVIER_MAX = 4.0          # plafond de la mise à l'échelle par la volatilité


def charger_fred(dossier: str | Path, ids: list[str], depuis: str = "1990") -> dict[str, pd.Series]:
    out = {}
    for s in ids:
        x = pd.read_csv(Path(dossier) / f"{s}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        x = pd.to_numeric(x, errors="coerce").dropna()
        x = x[x > 0][depuis:]
        if len(x) > 500:
            out[s] = x
    return out


def vol_ex_ante(close: pd.Series, span: int = 60) -> pd.Series:
    """Volatilité annualisée estimée avec les rendements <= t (EWMA)."""
    r = np.log(close).diff()
    return r.ewm(span=span, min_periods=span).std() * np.sqrt(ANN)


def rendements_strategie(close: pd.Series, signal: pd.Series, cout: float) -> pd.Series:
    """signal[t] ∈ [-1, 1] décidé à la clôture t ; rendement net de t à t+1."""
    r = close.pct_change().shift(-1)                       # rendement de t à t+1
    levier = (VOL_CIBLE / vol_ex_ante(close)).clip(upper=LEVIER_MAX)
    pos = (signal.reindex(close.index).fillna(0) * levier).fillna(0)
    frais = cout * pos.diff().abs().fillna(pos.abs())
    return (pos * r - frais).dropna()


def portefeuille(par_marche: dict[str, pd.Series]) -> pd.Series:
    """Poids égal entre les marchés disponibles chaque jour."""
    m = pd.DataFrame(par_marche)
    return m.mean(axis=1, skipna=True).dropna()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"n_jours": len(r)}
    eq = r.cumsum()
    sharpe = r.mean() / r.std() * np.sqrt(ANN)
    return {
        "n_jours": len(r),
        "rend_ann": r.mean() * ANN,
        "vol_ann": r.std() * np.sqrt(ANN),
        "sharpe": sharpe,
        "t_stat": sharpe * np.sqrt(len(r) / ANN),
        "dd_max": float((eq.cummax() - eq).max()),
    }


# ── Alphas ────────────────────────────────────────────────────────────────────

def alpha_tsmom(close: pd.Series, lookback: int = 252) -> pd.Series:
    """Momentum temporel : signe du rendement sur 12 mois (MOP 2012)."""
    return np.sign(close / close.shift(lookback) - 1)


def alpha_trend_mix(close: pd.Series) -> pd.Series:
    """Moyenne des signes à 1, 3 et 12 mois (variante courante chez les CTA)."""
    return sum(np.sign(close / close.shift(k) - 1) for k in (21, 63, 252)) / 3


def alpha_regime(reg: pd.DataFrame, seuil: float = 0.6) -> pd.Series:
    """+1 si P(haussier | passé) > seuil, -1 si P(baissier) > seuil, sinon 0."""
    s = pd.Series(0.0, index=reg.index)
    s[reg["p_bull"] > seuil] = 1.0
    s[reg["p_bear"] > seuil] = -1.0
    return s.where(reg["p_bull"].notna())


def filtre_regime(signal: pd.Series, state: pd.Series) -> pd.Series:
    """Garde le signal de tendance sauf quand le régime filtré dit l'inverse."""
    contre = (np.sign(signal) * state.reindex(signal.index)) < 0
    return signal.where(~contre, 0.0).where(state.reindex(signal.index).notna())
