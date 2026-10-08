"""
═══════════════════════════════════════════════════════════════════
TEST SCÉNARIO 4/5 — Données synthétiques à régime fort
═══════════════════════════════════════════════════════════════════

Lance ce script pour :
1. Générer des données avec TENDANCE FORTE (bull/bear distincts)
2. Réentraîner XGBoost + HMM sur ces données
3. Vérifier que >= 4 modèles votent dans la bonne direction

python tests/_4of5_scenario_test.py
"""
import sys
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deriv.ensemble_predictor import (
    EnsemblePredictor,
    HMMPredictor,
    XGBoostPredictor,
    KalmanMomentumPredictor,
    RSIDivergencePredictor,
    TrendStrengthPredictor,
    build_features,
)

# ── Générateur à régime fort ──────────────────────────────────────────
def generate_trending_ohlcv(n=400, trend="bull", noise=0.003):
    """Génère des données avec tendance CLAIRE (bull ou bear)."""
    rng = np.random.default_rng(42)
    dt_index = pd.date_range("2025-01-01", periods=n, freq="5min")

    if trend == "bull":
        # Tendance haussière : drift positif fort
        drift = 0.002  # 0.2% par bougie = forte tendance
    elif trend == "bear":
        # Tendance baissière : drift négatif fort
        drift = -0.002
    else:
        drift = 0.0  # range

    returns = rng.normal(drift, noise, size=n)
    close_prices = 100.0 * np.exp(np.cumsum(returns))

    # High/low réalistes
    high = close_prices * (1.0 + rng.uniform(0.001, 0.003, size=n))
    low = close_prices * (1.0 - rng.uniform(0.001, 0.003, size=n))
    open_prices = low + rng.uniform(0.2, 0.8, size=n) * (high - low)
    volume = rng.uniform(100.0, 500.0, size=n)

    return pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
    }, index=dt_index)


# ── Test scénario haussier ────────────────────────────────────────────
def test_bullish_scenario():
    print("\n" + "="*60)
    print("🟢 SCÉNARIO HAUSSIER (tendance forte)")
    print("="*60)

    df = generate_trending_ohlcv(400, trend="bull")

    ensemble = EnsemblePredictor()
    ensemble.train_all(df)

    result = ensemble.predict(df)

    print(f"Signal      : {result['signal']}")
    print(f"BUY votes   : {result['buy_votes']}")
    print(f"SELL votes  : {result['sell_votes']}")
    print(f"HOLD votes  : {result['hold_votes']}")
    print(f"Tradeable   : {result['tradeable']}")
    print()

    for name, detail in result["detail"].items():
        vote = detail['vote']
        emoji = "✅" if vote >= 0.7 else ("❌" if vote <= 0.3 else "⏸")
        print(f"  {emoji} {name:14s} vote={vote}")

    # Vérification : au moins 4 modèles doivent voter BUY
    assert result["buy_votes"] >= 4, f"ÉCHEC : seulement {result['buy_votes']} BUY (besoin de 4)"
    assert result["signal"] == "BUY", f"ÉCHEC : signal={result['signal']} (attendu BUY)"
    print("\n✅ SUCCÈS : 4+ modèles votent BUY")


# ── Test scénario baissier ────────────────────────────────────────────
def test_bearish_scenario():
    print("\n" + "="*60)
    print("🔴 SCÉNARIO BAISSIER (tendance forte)")
    print("="*60)

    df = generate_trending_ohlcv(400, trend="bear")

    ensemble = EnsemblePredictor()
    ensemble.train_all(df)

    result = ensemble.predict(df)

    print(f"Signal      : {result['signal']}")
    print(f"BUY votes   : {result['buy_votes']}")
    print(f"SELL votes  : {result['sell_votes']}")
    print(f"HOLD votes  : {result['hold_votes']}")
    print(f"Tradeable   : {result['tradeable']}")
    print()

    for name, detail in result["detail"].items():
        vote = detail['vote']
        emoji = "✅" if vote <= 0.3 else ("❌" if vote >= 0.7 else "⏸")
        print(f"  {emoji} {name:14s} vote={vote}")

    # Vérification : au moins 4 modèles doivent voter SELL
    assert result["sell_votes"] >= 4, f"ÉCHEC : seulement {result['sell_votes']} SELL (besoin de 4)"
    assert result["signal"] == "SELL", f"ÉCHEC : signal={result['signal']} (attendu SELL)"
    print("\n✅ SUCCÈS : 4+ modèles votent SELL")


# ── Test scénario range (doit donner HOLD) ─────────────────────────────
def test_range_scenario():
    print("\n" + "="*60)
    print("⏸ SCÉNARIO RANGE (pas de tendance)")
    print("="*60)

    df = generate_trending_ohlcv(400, trend="range", noise=0.004)

    ensemble = EnsemblePredictor()
    ensemble.train_all(df)

    result = ensemble.predict(df)

    print(f"Signal      : {result['signal']}")
    print(f"BUY votes   : {result['buy_votes']}")
    print(f"SELL votes  : {result['sell_votes']}")
    print(f"HOLD votes  : {result['hold_votes']}")
    print(f"Tradeable   : {result['tradeable']}")
    print()

    for name, detail in result["detail"].items():
        vote = detail['vote']
        emoji = "⏸" if 0.3 <= vote <= 0.7 else ("📈" if vote > 0.7 else "📉")
        print(f"  {emoji} {name:14s} vote={vote}")

    # En range, on s'attend à HOLD (pas 4 votes dans une direction)
    print(f"\nℹ️  En range : signal={result['signal']} (HOLD attendu ou votes dispersés)")


if __name__ == "__main__":
    try:
        test_bullish_scenario()
    except AssertionError as e:
        print(f"\n❌ {e}")

    try:
        test_bearish_scenario()
    except AssertionError as e:
        print(f"\n❌ {e}")

    try:
        test_range_scenario()
    except AssertionError as e:
        print(f"\n❌ {e}")

    print("\n" + "="*60)
    print("Tests terminés")
    print("="*60)