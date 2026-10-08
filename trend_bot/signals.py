"""Alpha : tendance multi-horizons (Moskowitz-Ooi-Pedersen 2012 ; Hurst-Ooi-Pedersen, AQR).

Toutes les fonctions n'utilisent que les clôtures <= t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252
HORIZONS = (21, 63, 252)       # 1, 3 et 12 mois de bourse
VOL_SPAN = 60


def trend_signal(close: pd.Series | pd.DataFrame, horizons=HORIZONS):
    """Moyenne des signes des rendements sur chaque horizon : valeurs dans [-1, 1].
    NaN tant que l'horizon le plus long n'est pas disponible."""
    sig = sum(np.sign(close / close.shift(k) - 1) for k in horizons) / len(horizons)
    return sig.where(close.shift(max(horizons)).notna())


def ex_ante_vol(close: pd.Series | pd.DataFrame, span: int = VOL_SPAN):
    """Volatilité annualisée estimée par EWMA sur les rendements log <= t."""
    r = np.log(close).diff()
    return r.ewm(span=span, min_periods=span).std() * np.sqrt(ANN)


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
