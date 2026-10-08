"""Construction : signaux → poids (volatilité cible du portefeuille) → lots.

1. Chaque marché reçoit signal × (vol_cible / vol_marché) / N (risque égal).
2. Le tout est remis à l'échelle pour que la volatilité ex-ante du PORTEFEUILLE
   (covariance sur 1 an) vaille `port_vol` (10 %/an comme AQR), levier brut plafonné.
3. Poids → lots via la valeur d'un lot en devise du compte ; zone neutre pour ne pas
   payer de frais sur de petits ajustements.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_bot.signals import ANN

PORT_VOL = 0.10        # volatilité annuelle visée pour le portefeuille
MAX_GROSS = 3.0        # somme des |poids| (notionnel / équité) maximale
COV_WIN = 252
BUFFER = 0.25          # pas d'ordre si l'écart < 25 % de la cible (même sens)


def target_weights(signals: pd.Series, vols: pd.Series, returns: pd.DataFrame,
                   port_vol: float = PORT_VOL, max_gross: float = MAX_GROSS) -> pd.Series:
    """Poids cibles (fraction de l'équité, signés). `returns` : rendements journaliers
    passés (lignes <= t) servant à la covariance."""
    ok = signals.notna() & vols.notna() & (vols > 0)
    w = pd.Series(0.0, index=signals.index)
    if not ok.any():
        return w
    raw = signals[ok] / vols[ok] / ok.sum()
    if (raw == 0).all():
        return w
    R = returns[raw.index].iloc[-COV_WIN:].dropna(how="all").fillna(0.0)
    cov = R.cov().to_numpy() * ANN
    pv = float(np.sqrt(max(raw.to_numpy() @ cov @ raw.to_numpy(), 1e-12)))
    w[raw.index] = raw * (port_vol / pv)
    gross = w.abs().sum()
    if gross > max_gross:
        w *= max_gross / gross
    return w


@dataclass
class SymbolSpec:
    lot_value: float      # valeur notionnelle d'1 lot en devise du compte
    volume_min: float
    volume_max: float
    volume_step: float


def weight_to_lots(weight: float, equity: float, spec: SymbolSpec) -> float:
    raw = weight * equity / spec.lot_value
    lots = np.floor(abs(raw) / spec.volume_step + 1e-9) * spec.volume_step
    if lots < spec.volume_min:
        return 0.0
    return float(np.sign(raw) * min(lots, spec.volume_max))


def needs_trade(current: float, target: float, step: float, buffer: float = BUFFER) -> bool:
    """Zone neutre : on ne trade que si le sens change ou si l'écart est significatif.
    `step` = plus petit ajustement possible (pas de volume, ou ~0 en poids)."""
    if np.sign(current) != np.sign(target):
        return abs(target - current) >= step / 2
    return abs(target - current) > max(buffer * abs(target), step / 2)
