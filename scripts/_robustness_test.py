"""
Test de robustesse 4/5 — scénarios difficiles.
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, r'c:\BOT-TRADING')

def make_trend(n=600, trend=0.002, noise=0.003, seed=42):
    rng = np.random.RandomState(seed)
    ret = rng.normal(trend, noise, n)
    close = 100 * np.exp(np.cumsum(ret))
    high = close * (1 + abs(rng.normal(0, noise/2, n)))
    low = close * (1 - abs(rng.normal(0, noise/2, n)))
    open_ = np.r_[close[0], close[:-1]]
    vol = rng.randint(10, 100, n)
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})
    return df

def main():
    scenarios = [
        ("FORT_HAUT",    make_trend(trend=0.003, noise=0.002, seed=1)),
        ("FORT_BAS",     make_trend(trend=-0.003, noise=0.002, seed=2)),
        ("RANGE_PUR",    make_trend(trend=0.0, noise=0.005, seed=3)),
        ("HAUT_FAIBLE",  make_trend(trend=0.0005, noise=0.004, seed=4)),  # doit être HOLD
        ("BAS_FAIBLE",   make_trend(trend=-0.0005, noise=0.004, seed=5)), # doit être HOLD
        ("BRUIT",        make_trend(trend=0.0, noise=0.008, seed=6)),
    ]

    from deriv.ensemble_predictor import EnsemblePredictor
    ens = EnsemblePredictor()
    # Entraînement sur fort haut + fort bas + range pur
    df_all = pd.concat([scenarios[0][1], scenarios[1][1], scenarios[2][1]], ignore_index=True)
    ens.train_all(df_all)

    results = []
    for name, df in scenarios:
        res = ens.predict(df)
        results.append((name, res))
        print(f"\n--- {name} ---")
        print(f"Signal: {res['signal']} | BUY:{res['buy_votes']} SELL:{res['sell_votes']} HOLD:{res['hold_votes']}")
        for k, v in res['detail'].items():
            print(f"  {k:15s} -> {v}")

    print("\n" + "="*60)
    print("RÉSUMÉ:")
    for name, res in results:
        print(f"  {name:15s} -> {res['signal']:5s} (B:{res['buy_votes']} S:{res['sell_votes']} H:{res['hold_votes']})")

if __name__ == "__main__":
    main()