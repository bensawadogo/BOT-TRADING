"""Alpha : tendance multi-horizons (Moskowitz-Ooi-Pedersen 2012 ; Hurst-Ooi-Pedersen, AQR).

Toutes les fonctions n'utilisent que les clôtures <= t.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANN = 252
HORIZONS = (21, 63, 252)       # 1, 3 et 12 mois de bourse
VOL_SPAN = 60
# Prévisions continues de Carver (Systematic Trading) : (EWMA rapide, EWMA lente, échelle qui
# ramène la prévision moyenne à 10). Fixées à l'avance, jamais réglées sur nos données.
EWMAC_SPEEDS = ((16, 64, 3.75), (32, 128, 2.65), (64, 256, 1.87))
FORECAST_CAP = 20.0


def weekdays_only(close: pd.Series | pd.DataFrame):
    """Ne garde que les clôtures du lundi au vendredi, le calendrier du backtest (FRED).
    Deriv MT5 crée une bougie D1 le dimanche sur le forex (1 bougie sur 6) et les cryptos
    cotent 7 j/7 : sans ce filtre, les horizons en bougies et ANN = 252 ne correspondent
    plus à ce qui a été testé. Le mouvement du week-end est porté par le lundi."""
    return close[close.index.day_of_week < 5]


def trend_signal(close: pd.Series | pd.DataFrame, horizons=HORIZONS):
    """Moyenne des signes des rendements sur chaque horizon : valeurs dans [-1, 1].
    NaN tant que l'horizon le plus long n'est pas disponible."""
    sig = sum(np.sign(close / close.shift(k) - 1) for k in horizons) / len(horizons)
    return sig.where(close.shift(max(horizons)).notna())


def ex_ante_vol(close: pd.Series | pd.DataFrame, span: int = VOL_SPAN):
    """Volatilité annualisée estimée par EWMA sur les rendements log <= t."""
    r = np.log(close).diff()
    return r.ewm(span=span, min_periods=span).std() * np.sqrt(ANN)


def ewmac(close: pd.Series, fast: int, slow: int, scalar: float) -> pd.Series:
    """Prévision de tendance continue dans [-1, 1] : (EWMA rapide - EWMA lente) rapporté à
    l'écart-type quotidien du prix, mis à l'échelle de Carver puis plafonné à ±20 (÷ 20).
    NaN tant que `slow` + 1 clôtures ne sont pas disponibles."""
    px_vol = close * ex_ante_vol(close) / np.sqrt(ANN)
    raw = (close.ewm(span=fast, adjust=False).mean()
           - close.ewm(span=slow, adjust=False).mean()) / px_vol
    return (raw * scalar / FORECAST_CAP).clip(-1, 1).where(close.shift(slow).notna())


def trend_forecast(close: pd.Series, speeds=EWMAC_SPEEDS, desaccord: float = 1.0) -> pd.Series:
    """Moyenne à poids égaux des prévisions de plusieurs vitesses (rien d'optimisé).
    `desaccord` < 1 : quand la vitesse la plus rapide et la plus lente sont de sens opposés
    (retournement probable, Goulding-Harvey-Mazzoleni, JFE 2023), la prévision est
    multipliée par ce facteur."""
    f = [ewmac(close, a, b, k) for a, b, k in speeds]
    combine = sum(f) / len(f)
    if desaccord != 1.0:
        contredit = np.sign(f[0]) * np.sign(f[-1]) < 0
        combine = combine.where(~contredit, combine * desaccord)
    return combine.where(f[-1].notna())


@dataclass(frozen=True)
class Strategie:
    """Les trois rôles : direction (signal), confiance (désaccord), risque (régime).
    Les valeurs par défaut sont le bot d'origine ; backtest et live partagent cet objet."""
    signal: str = "signe"          # "signe" : signes 1/3/12 mois ; "ewmac" : prévisions continues
    desaccord: float = 1.0         # facteur appliqué si rapide et lent se contredisent (ewmac)
    regime_facteur: float = 0.0    # part du signal gardée quand le régime le contredit (0 = coupe)


def market_signal(close: pd.Series, strat: Strategie = Strategie()) -> pd.Series:
    if strat.signal == "ewmac":
        return trend_forecast(close, desaccord=strat.desaccord)
    return trend_signal(close)


def per_market(func, closes: pd.DataFrame, max_gap: int = 10) -> pd.DataFrame:
    """Applique `func` à chaque marché sur SA série (sans les jours où il ne cote pas),
    puis reporte la dernière valeur connue sur le calendrier commun (au plus `max_gap` j).
    Évite les NaN créés par les jours fériés propres à chaque marché."""
    cols = {c: func(closes[c].dropna()) for c in closes}
    return pd.DataFrame(cols).reindex(closes.index).ffill(limit=max_gap)


def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Rendements simples calculés sur la série propre de chaque marché : le rendement
    qui enjambe un jour férié est porté par le jour de reprise (NaN les jours fermés)."""
    return pd.DataFrame({c: closes[c].dropna().pct_change() for c in closes}
                        ).reindex(closes.index)


def regime_features(close: pd.Series, vol_win: int = 20, mom_win: int = 10) -> pd.DataFrame:
    """Entrées des modèles de régime : rendement log, volatilité 20 j, momentum 10 j."""
    r = np.log(close).diff()
    return pd.DataFrame({
        "ret": r,
        "vol": r.rolling(vol_win).std(),
        "mom": r.rolling(mom_win).sum(),
    }, index=close.index)
