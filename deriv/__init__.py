"""Module Deriv pour l'automatisation de trading d'indices synthétiques."""

from .bot import DerivBot
from .client import DerivClient, DerivConnection
from .risk_manager import RiskManager, StakeDecision, TradeResult
from .data_collector import DerivDataCollector
from .ensemble_predictor import EnsemblePredictor, build_features
from .bot_executor import DerivBotExecutor

__all__ = [
    "DerivBot",
    "DerivClient",
    "DerivConnection",
    "RiskManager",
    "StakeDecision",
    "TradeResult",
    "DerivDataCollector",
    "EnsemblePredictor",
    "build_features",
    "DerivBotExecutor",
]
