"""Bot de suivi de tendance multi-marchés (journalier).

Chaîne, à la manière du framework LEAN :
    signals (alpha)  →  portfolio (volatilité cible, lots)  →  risk (régime, coupe-circuit)
    →  brokers (papier ou MetaTrader 5)  ;  runner orchestre un cycle par jour.

Validation : docs/reports/SYSTEME_REGIME_FIBO_2026-10-08.md et trend_bot/backtest.py.
"""
