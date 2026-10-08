"""Mesure empirique des crashs CRASH500 M1 — calibre le seuil du CrashGuard.

Sortie (crash_stats.txt) :
  - distribution des rendements M1 (percentiles),
  - part de bougies de crash pour plusieurs seuils,
  - statistique de streaks haussiers AVANT crash (pour le modèle),
  - intervalle moyen entre crashs (pour la prior de Poisson).
"""
import sys, asyncio
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from deriv.data_collector import DerivDataCollector


def streak_stats(close: pd.Series, seuil: float):
    """Streak haussier courant avant chaque crash vs avant bougie normale."""
    ret = close.pct_change()
    crash = ret <= -seuil
    # streak = nb de bougies haussières consécutives qui précèdent
    up = ret > 0
    streak = up.groupby((~up).cumsum()).cumsum()  # compte les up consécutifs
    s_crash = streak[crash].dropna()
    s_norm = streak[~crash].dropna()
    return s_crash, s_norm


async def main():
    c = DerivDataCollector('CRASH500', 60)
    await c.connect()
    df = await c.get_candle_history_full(target_count=5000)
    await c.close()

    close = df['close']
    ret = close.pct_change().dropna()

    lignes = [f"bougies={len(df)}  rendements={len(ret)}"]
    lignes.append("percentiles |ret| : " + " ".join(
        f"p{q}={abs(ret).quantile(q/100):.4%}" for q in (50, 75, 90, 95, 99)))
    lignes.append("queue négative    : " + " ".join(
        f"p{q}={ret.quantile(q/100):.4%}" for q in (1, 5, 10, 25)))
    lignes.append(f"taux UP global    : {(ret > 0).mean():.4f}")

    for seuil in (0.001, 0.002, 0.003, 0.005, 0.01):
        n_crash = int((ret <= -seuil).sum())
        lignes.append(
            f"seuil crash -{seuil:.1%} : {n_crash} bougies "
            f"({n_crash/len(ret):.2%})  | amorties (tau={int(1/max(n_crash/len(ret),1e-9)) if n_crash else 0})"
        )

    # Meilleur seuil candidat ≈ p2 des rendements (2% de crashs)
    seuil = float(-ret.quantile(0.02))
    lignes.append(f"seuil retenu (p2) : -{seuil:.4%}")

    s_crash, s_norm = streak_stats(close, seuil)
    if len(s_crash):
        lignes.append(
            f"streak AVANT crash  : moyenne={s_crash.mean():.1f} "
            f"med={s_crash.median():.0f} p75={s_crash.quantile(0.75):.0f} (n={len(s_crash)})"
        )
    if len(s_norm):
        lignes.append(
            f"streak AVANT normal : moyenne={s_norm.mean():.1f} "
            f"med={s_norm.median():.0f} p75={s_norm.quantile(0.75):.0f} (n={len(s_norm)})"
        )

    # Intervalle entre crashs (en bougies)
    idx_crash = np.flatnonzero((ret <= -seuil).values)
    if len(idx_crash) > 1:
        gaps = np.diff(idx_crash)
        lignes.append(
            f"ecart entre crashs : moyenne={gaps.mean():.1f} med={np.median(gaps):.0f} "
            f"max={gaps.max()} bougies"
        )

    with open('crash_stats.txt', 'w', encoding='utf-8') as f:
        f.write("\n".join(lignes) + "\n")
    print("\n".join(lignes))


asyncio.run(main())