"""Données du backtest : clôtures journalières FRED (gratuites, forex depuis 1971).

Téléchargement : python scripts/fetch_fred.py data/fred
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

FRED = {  # identifiant: coût par unité de rotation (fraction)
    "DEXUSEU": 1e-4, "DEXUSUK": 1e-4, "DEXJPUS": 1e-4, "DEXUSAL": 1e-4, "DEXCAUS": 1e-4,
    "DEXSZUS": 1e-4, "DEXUSNZ": 2e-4, "DEXMXUS": 3e-4, "DEXNOUS": 3e-4, "DEXSDUS": 3e-4,
    "DCOILWTICO": 5e-4, "DCOILBRENTEU": 5e-4, "NASDAQCOM": 2e-4, "DHHNGSP": 1e-3,
}

# Série FRED → marché du bot (trend_bot.config.UNIVERSE), même sens de cotation.
MARCHE = {
    "DEXUSEU": "EURUSD", "DEXUSUK": "GBPUSD", "DEXJPUS": "USDJPY", "DEXUSAL": "AUDUSD",
    "DEXCAUS": "USDCAD", "DEXSZUS": "USDCHF", "DEXUSNZ": "NZDUSD", "DEXMXUS": "USDMXN",
    "DEXNOUS": "USDNOK", "DEXSDUS": "USDSEK", "DCOILWTICO": "USOIL", "DCOILBRENTEU": "UKOIL",
    "NASDAQCOM": "NAS100", "DHHNGSP": "NGAS",
}


def charger_fred(dossier: str | Path, ids: list[str], depuis: str = "1990") -> dict[str, pd.Series]:
    out = {}
    for s in ids:
        x = pd.read_csv(Path(dossier) / f"{s}.csv", index_col=0, parse_dates=True).iloc[:, 0]
        x = pd.to_numeric(x, errors="coerce").dropna()
        x = x[x > 0][depuis:]
        if len(x) > 500:
            out[s] = x
    return out


# Frais d'administration du compte Deriv MT5 SANS SWAP (grille publiée par Deriv, relevée le
# 06/11/2026) : USD par lot et par nuit, (position longue, position courte), dus APRÈS
# 15 jours de détention d'une position sur un instrument financier.
# https://blog.deriv.com/blog/admin-fees-for-deriv-mt5-swap-free-accounts
# Valeur notionnelle d'un lot : 100 000 unités de la devise de base (forex), 1 × l'indice (Nasdaq).
ADMIN_SANS_SWAP = {   # série FRED → (frais long, frais court, devise de base ≠ USD ?)
    "DEXUSEU": (8.80, 0.0, True), "DEXUSUK": (1.87, 2.42, True), "DEXJPUS": (0.0, 14.89, False),
    "DEXUSAL": (0.0, 2.86, True), "DEXCAUS": (0.0, 7.20685, False),
    "DEXSZUS": (0.0, 12.0571, False), "DEXUSNZ": (7.12, 2.17, True),
    "DEXMXUS": (8.94347, 0.0, False), "DEXNOUS": (6.59237, 4.07882, False),
    "DEXSDUS": (0.02356, 11.03902, False), "NASDAQCOM": (4.89611, 0.0, None),
}
GRACE_JOURS = 15


def taux_admin(prix: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Frais d'administration en fraction annuelle du notionnel : (long, court).
    `prix` : dernier cours de chaque série FRED (notionnel d'un lot en USD)."""
    out = {}
    for sid, (fl, fc, base_non_usd) in ADMIN_SANS_SWAP.items():
        notionnel = prix[sid] if base_non_usd is None else 1e5 * (prix[sid] if base_non_usd else 1.0)
        out[sid] = (fl * 365 / notionnel, fc * 365 / notionnel)
    return out
