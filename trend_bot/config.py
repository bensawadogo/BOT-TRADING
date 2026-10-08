"""Paramètres du bot de tendance (surcharge possible via .env)."""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]   # dossier du projet, quel que soit le cwd

# Marchés : nom logique → noms possibles chez le courtier (le premier trouvé est pris).
# Noms usuels Deriv MT5 / autres courtiers ; vérifier dans MT5 (Ctrl+M) si besoin.
UNIVERSE: dict[str, list[str]] = {
    "EURUSD": ["EURUSD"], "GBPUSD": ["GBPUSD"], "USDJPY": ["USDJPY"],
    "AUDUSD": ["AUDUSD"], "USDCAD": ["USDCAD"], "USDCHF": ["USDCHF"],
    "NZDUSD": ["NZDUSD"], "EURJPY": ["EURJPY"], "GBPJPY": ["GBPJPY"],
    "XAUUSD": ["XAUUSD", "GOLD"], "XAGUSD": ["XAGUSD", "SILVER"],
    "NAS100": ["US Tech 100", "USTEC", "NAS100", "US100"],
    "SPX500": ["US SP 500", "US500", "SPX500", "US 500"],
    "US30": ["Wall Street 30", "US30", "DJ30"],
    "GER40": ["Germany 40", "GER40", "DE40", "Germany 30", "GER30"],
    "UK100": ["UK 100", "UK100", "FTSE100"],
    "JPN225": ["Japan 225", "JP225", "JPN225"],
    "BTCUSD": ["BTCUSD"], "ETHUSD": ["ETHUSD"],
}


def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def symbols() -> list[str]:
    """Sous-ensemble choisi via TREND_SYMBOLS=EURUSD,XAUUSD,... (défaut : tous)."""
    raw = os.getenv("TREND_SYMBOLS", "").strip()
    return [s.strip() for s in raw.split(",") if s.strip()] if raw else list(UNIVERSE)


PORT_VOL = _f("TREND_PORT_VOL", 0.10)       # volatilité annuelle visée
MAX_GROSS = _f("TREND_MAX_GROSS", 3.0)      # levier brut max (notionnel / équité)
SOFT_DD = _f("TREND_SOFT_DD", 0.10)         # drawdown → exposition / 2
HARD_DD = _f("TREND_HARD_DD", 0.20)         # drawdown → tout couper, reset manuel
RUN_HOUR_UTC = int(_f("TREND_RUN_HOUR_UTC", 7))    # marchés ouverts, bougie de la veille terminée
HISTORY_DAYS = 1300                          # ≥ 1000 (régime) + marge
MAGIC = 770077                               # identifie les positions du bot dans MT5
ALLOW_REAL = os.getenv("MT5_ALLOW_REAL", "0") == "1"
STATE_PATH = os.getenv("TREND_STATE", str(_ROOT / "data" / "trend_state.json"))
JOURNAL_PATH = os.getenv("TREND_JOURNAL", str(_ROOT / "data" / "trend_journal.csv"))
