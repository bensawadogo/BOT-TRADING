"""Constantes partagées du projet Deriv Bot.

Source unique de vérité pour toutes les constantes configurables.
Les autres modules doivent IMPORTER depuis ce fichier.
"""
from __future__ import annotations
import os

BREAKEVEN = 0.556
BREAKEVEN_WINRATE = 55.6
MIN_VOTES_TO_TRADE = 4
PROBA_BUY = 0.8
PROBA_SELL = 0.2
PROBA_NEUTRAL = 0.5
VOTE_UP_THRESHOLD = 0.7
VOTE_DOWN_THRESHOLD = 0.3
VOTE_FROM_PROBA_UP = 0.65
VOTE_FROM_PROBA_DOWN = 0.35
CRASH_SEUIL_FRAC = 0.001537
SPREAD_CRASH = 0.2
SPREAD_BOOM = 0.2
SPREAD_DEFAULT = 0.2
MAX_CONSECUTIVE = 3
MAX_STAKE_PCT = 0.02
MAX_DAILY_LOSS = 0.10
COOLDOWN_MINUTES = 15
KELLY_FRACTION = 0.25
WFO_TRAIN = {60: 400, 300: 300, 600: 300, 1440: 200}
WFO_TEST = {60: 100, 300: 75, 600: 50, 1440: 50}
WFO_EMBARGO = {60: 30, 300: 10, 600: 5, 1440: 5}
DEFAULT_SYMBOL = "CRASH500"
DEFAULT_GRANULARITY = 60
DEFAULT_TRAIN_COUNT = 3000
DEFAULT_DURATION_MIN = 240
DEFAULT_APP_ID = "1089"
TOKEN_PLACEHOLDER = "REMPLACE_PAR_TON_TOKEN"
MIN_TRADES_DEMO = 200
MAX_STD_DEMO = 0.15

# Sécurité — désérialisation safe (anti-RCE pickle)

import pickle as _pickle

_SAFE_PREFIXES = (
    "collections", "datetime", "enum", "functools",
    "numpy", "pandas", "sklearn", "hmmlearn", "xgboost",
    "scipy", "tensorflow", "keras", "river", "ta", "typing",
)

_SAFE_BUILTINS = frozenset({
    "int", "float", "str", "bool", "list", "dict", "tuple", "set",
    "frozenset", "complex", "bytes", "bytearray", "range", "slice",
    "type", "NoneType", "ellipsis", "NotImplemented",
})

_DANGEROUS_BUILTINS = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "exit", "quit", "license", "help", "dir", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "__import__",
})

class SafeUnpickler(_pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in _SAFE_BUILTINS and name not in _DANGEROUS_BUILTINS:
            return super().find_class(module, name)
        if module.startswith(_SAFE_PREFIXES):
            return super().find_class(module, name)
        raise _pickle.UnpicklingError(f"Refus : {module}.{name}")

def safe_pickle_load(f):
    return SafeUnpickler(f).load()

def safe_pickle_dump(data, f):
    import pickle as _pickle
    _pickle.dump(data, f)

class Config:
    "Configuration centralisee depuis les variables d environnement"
    APP_ID = os.getenv("DERIV_APP_ID", DEFAULT_APP_ID)
    SYMBOL = os.getenv("DERIV_SYMBOL", DEFAULT_SYMBOL)
    DURATION = int(os.getenv("DERIV_DUR", str(DEFAULT_DURATION_MIN)))
    ACCOUNT_TYPE = os.getenv("DERIV_ACCOUNT_TYPE", "demo")
    API_TOKEN = os.getenv("DERIV_API_TOKEN", "")
    AUTO_MAX_STAKE = float(os.getenv("DERIV_AUTO_MAX_STAKE", "5"))
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
    CAPITAL_USD = float(os.getenv("CAPITAL_USD", "150"))
    CYCLE_SEC = int(os.getenv("CYCLE_SEC", "60"))
    RUN_NETWORK_TESTS = os.getenv("RUN_NETWORK_TESTS", "0") == "1"
    ALLOW_REAL = os.getenv("DERIV_ALLOW_REAL", "0") == "1"
    GRANULARITY = int(os.getenv("DERIV_GRANULARITY", str(DEFAULT_GRANULARITY)))
    MAX_RISK_PCT = float(os.getenv("MAX_RISK_PCT", str(MAX_STAKE_PCT)))
    DAILY_STOP_PCT = float(os.getenv("DAILY_STOP_PCT", str(MAX_DAILY_LOSS)))

    MANUAL_THRESHOLD = float(os.getenv("DERIV_MANUAL_THRESHOLD", "5"))
    MAX_CONSECUTIVE = int(os.getenv("MAX_CONSECUTIVE", "3"))
    AUTO_STAKE = float(os.getenv("AUTO_STAKE", "2.0"))
    JOURNAL_DB = os.getenv("JOURNAL_DB", "")

    @classmethod
    def validate(cls):
        errs = []
        if cls.CAPITAL_USD <= 0:
            errs.append("CAPITAL_USD must be positive")
        if not cls.SYMBOL:
            errs.append("DERIV_SYMBOL required")
        return errs

    @classmethod
    def summary(cls):
        return dict(symbol=cls.SYMBOL, app_id=cls.APP_ID, account_type=cls.ACCOUNT_TYPE, capital=cls.CAPITAL_USD, has_telegram=bool(cls.TELEGRAM_BOT_TOKEN))

