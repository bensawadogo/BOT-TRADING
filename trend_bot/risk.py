"""Risque : filtre de régime (modèle à sauts) et coupe-circuit de drawdown.

Le régime ne sert PAS à choisir le sens (sans avantage au backtest) : il coupe la
position quand il contredit la tendance. Aux coûts réels de Deriv, sur 1985-2026, il réduit
le drawdown maximal de 42 % à 30 % et le Sharpe passe de 0,22 à 0,28 ; depuis 1996 il évite
des pertes (−0,03 → +0,06) sans créer de rendement (VERIFICATION_LOGIQUE_2026-11-10.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trend_bot.signals import regime_features

TRAIN_LEN = 1000           # jours d'historique pour entraîner le modèle à sauts
JUMP_PENALTY = 50.0


def _fit_jump(X: np.ndarray, jump_penalty: float):
    """Entraîne le modèle à sauts sur X (normalisé avec X seul). Renvoie (modèle, mu, sd)."""
    from jumpmodels.jump import JumpModel

    mu, sd = X.mean(0), X.std(0) + 1e-12
    jm = JumpModel(n_components=3, jump_penalty=jump_penalty, random_state=0, n_init=3)
    jm.fit((X - mu) / sd, ret_ser=X[:, 0], sort_by="cumret")   # état 0 = le plus haussier
    return jm, mu, sd


def _label(states: np.ndarray) -> np.ndarray:
    return np.where(states == 0, 1, np.where(states == 2, -1, 0))


def _refits(dates: pd.DatetimeIndex, train_len: int) -> np.ndarray:
    """Positions des ré-entraînements : première bougie de chaque mois, avec au moins
    `train_len` bougies avant. Ancrées au calendrier, elles ne dépendent pas du début de
    l'historique : le live et le backtest ré-entraînent aux mêmes dates."""
    mois = dates.to_period("M")
    debuts = np.flatnonzero(np.r_[True, mois[1:] != mois[:-1]])
    return debuts[debuts >= train_len]


def _states(X: np.ndarray, k: int, fin: int, train_len: int, jump_penalty: float) -> np.ndarray:
    """Entraîne sur X[k - train_len:k], puis états en ligne (causaux) de k à fin - 1."""
    jm, mu, sd = _fit_jump(X[k - train_len:k], jump_penalty)
    return _label(np.asarray(jm.predict_online((X[k - train_len:fin] - mu) / sd)))[train_len:]


def regime_now(close: pd.Series, train_len: int = TRAIN_LEN,
               jump_penalty: float = JUMP_PENALTY) -> int | None:
    """État -1 / 0 / +1 de la dernière clôture, calculé exactement comme `regime_series`
    (même ré-entraînement mensuel). None si l'historique est trop court (pas de filtre)."""
    F = regime_features(close).dropna()
    ks = _refits(F.index, train_len)
    if not len(ks):
        return None
    return int(_states(F.to_numpy(float), ks[-1], len(F), train_len, jump_penalty)[-1])


def regime_series(close: pd.Series, train_len: int = TRAIN_LEN,
                  jump_penalty: float = JUMP_PENALTY) -> pd.Series:
    """Version backtest de `regime_now` : un modèle par mois, entraîné sur les `train_len`
    jours précédant le 1er jour du mois, puis états en ligne (causal) jusqu'au mois suivant."""
    F = regime_features(close).dropna()
    X = F.to_numpy(float)
    out = pd.Series(np.nan, index=close.index)
    ks = _refits(F.index, train_len)
    for k, fin in zip(ks, [*ks[1:], len(F)]):
        out.loc[F.index[k:fin]] = _states(X, k, fin, train_len, jump_penalty)
    return out


def apply_regime(signal: float, state: int | None, facteur: float = 0.0) -> float:
    """Quand le régime dit l'inverse de la tendance, ne garde que `facteur` du signal
    (0 : annulé, 0,5 : réduit de moitié)."""
    if state is None or np.isnan(signal):
        return signal
    return facteur * signal if np.sign(signal) * state < 0 else signal


@dataclass
class KillSwitch:
    """Drawdown depuis le plus haut de l'équité :
    au-delà de `soft`, exposition divisée par 2 ; au-delà de `hard`, tout est coupé
    et le bot reste à plat jusqu'à une remise à zéro MANUELLE (reset)."""
    soft: float = 0.10
    hard: float = 0.20
    peak: float = 0.0
    tripped: bool = False

    def update(self, equity: float) -> float:
        """Renvoie le multiplicateur d'exposition (1, 0,5 ou 0)."""
        self.peak = max(self.peak, equity)
        dd = 1 - equity / self.peak if self.peak > 0 else 0.0
        if dd >= self.hard:
            self.tripped = True
        if self.tripped:
            return 0.0
        return 0.5 if dd >= self.soft else 1.0

    def reset(self, equity: float) -> None:
        self.peak, self.tripped = equity, False
