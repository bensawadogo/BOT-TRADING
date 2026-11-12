"""Paramètres du bot de tendance (surcharge possible via .env)."""
from __future__ import annotations

import os
from pathlib import Path

from trend_bot.signals import Strategie

_ROOT = Path(__file__).resolve().parents[1]   # dossier du projet, quel que soit le cwd

# Panier par défaut : marchés du backtest (trend_bot.data.FRED) tradables sur Deriv MT5.
# Nom logique → noms possibles chez le courtier (le premier trouvé est pris).
UNIVERSE: dict[str, list[str]] = {
    "EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"], "USDJPY": ["USDJPY"],
    "AUDUSD": ["AUDUSD"], "USDCAD": ["USDCAD"], "USDCHF": ["USDCHF"],
    "NZDUSD": ["NZDUSD"], "USDMXN": ["USDMXN"], "USDNOK": ["USDNOK"], "USDSEK": ["USDSEK"],
    "NAS100": ["US Tech 100", "USTEC", "NAS100", "US100"],
}

# Backtestés mais EXCLUS : swaps relevés sur Deriv MT5 le 02/11/2026 ≈ −60 à −75 %/an
# (pétrole) et jusqu'à −224 %/an (gaz acheté) ; avec eux le bot perd de l'argent
# (docs/reports/TREND_BOT_COUTS_REELS_2026-11-02.md). Sur demande seulement (TREND_SYMBOLS).
ENERGIES: dict[str, list[str]] = {
    "USOIL": ["US Oil", "USOIL", "WTI", "XTIUSD"],
    "UKOIL": ["UK Brent Oil", "UKOIL", "BRENT", "XBRUSD"],
    "NGAS": ["NGAS", "NATGAS", "XNGUSD"],
}

# Balayés le 10/11/2026 avec leurs coûts réels : AUCUN avantage de tendance mesuré (crypto = bêta,
# swaps -15 à -25 %/an ; docs/reports/SCAN_MARCHES_2026-11-10.md). Sur demande seulement
# (TREND_SYMBOLS), avertissement à chaque cycle.
NON_VALIDES: dict[str, list[str]] = {
    "EURJPY": ["EURJPY"], "GBPJPY": ["GBPJPY"],
    "XAUUSD": ["XAUUSD", "GOLD"], "XAGUSD": ["XAGUSD", "SILVER"],
    "SPX500": ["US SP 500", "US500", "SPX500", "US 500"],
    "US30": ["Wall Street 30", "US30", "DJ30"],
    "GER40": ["Germany 40", "GER40", "DE40", "Germany 30", "GER30"],
    "UK100": ["UK 100", "UK100", "FTSE100"],
    "JPN225": ["Japan 225", "JP225", "JPN225"],
    "BTCUSD": ["BTCUSD"], "ETHUSD": ["ETHUSD"],
}
ALIASES = {**UNIVERSE, **ENERGIES, **NON_VALIDES}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def symbols() -> list[str]:
    """Marchés choisis via TREND_SYMBOLS=EURUSD,NAS100,... (défaut : UNIVERSE)."""
    raw = os.getenv("TREND_SYMBOLS", "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else list(UNIVERSE)


# Stratégie en production : celle qui a passé le protocole du 06/11/2026
# (docs/reports/ESSAIS_E0_E4_2026-11-06.md). Backtest et live lisent ce même objet.
STRATEGIE = Strategie()

PORT_VOL = _f("TREND_PORT_VOL", 0.10)       # volatilité annuelle visée
MAX_GROSS = _f("TREND_MAX_GROSS", 3.0)      # levier brut max (notionnel / équité)
SOFT_DD = _f("TREND_SOFT_DD", 0.10)         # drawdown → exposition / 2
HARD_DD = _f("TREND_HARD_DD", 0.20)         # drawdown → tout couper, reset manuel
RUN_HOUR_UTC = int(_f("TREND_RUN_HOUR_UTC", 7))    # marchés ouverts, bougie de la veille terminée
RUN_END_UTC = int(_f("TREND_RUN_END_UTC", 20))     # plus d'ordre après : marchés fermés ou illiquides
NEWS_FILTER = os.getenv("TREND_NEWS", "1") == "1"  # reporter les ordres autour des annonces
NEWS_PATH = os.getenv("TREND_NEWS_CACHE", str(_ROOT / "data" / "news_cache.json"))
HISTORY_DAYS = 2000                          # bougies D1 : ≥ 1000 jours ouvrés (régime) + marge
MAGIC = 770077                               # identifie les positions du bot dans MT5
ALLOW_REAL = os.getenv("MT5_ALLOW_REAL", "0") == "1"
STATE_PATH = os.getenv("TREND_STATE", str(_ROOT / "data" / "trend_state.json"))
JOURNAL_PATH = os.getenv("TREND_JOURNAL", str(_ROOT / "data" / "trend_journal.csv"))
