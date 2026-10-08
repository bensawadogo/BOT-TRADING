"""
Test d'entraînement + vote 4/5 sur scénario à régime fort.
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
    # Scénario 1 : forte hausse
    df_up = make_trend(trend=0.003, noise=0.002)
    # Scénario 2 : forte baisse
    df_down = make_trend(trend=-0.003, noise=0.002)
    # Scénario 3 : range
    df_range = make_trend(trend=0.0, noise=0.004)

    from deriv.ensemble_predictor import EnsemblePredictor
    ens = EnsemblePredictor()
    # Entraînement sur les 3 scénarios concaténés
    df_all = pd.concat([df_up, df_down, df_range], ignore_index=True)
    print(f"Entraînement sur {len(df_all)} bougies...")
    ens.train_all(df_all)

    # Test sur chaque scénario
    for name, df in [("HAUSSIER", df_up), ("BAISSER", df_down), ("RANGE", df_range)]:
        res = ens.predict(df)
        print(f"\n--- {name} ---")
        print(f"Signal: {res['signal']} | BUY:{res['buy_votes']} SELL:{res['sell_votes']} HOLD:{res['hold_votes']}")
        for k, v in res['detail'].items():
            print(f"  {k:15s} -> {v}")

if __name__ == "__main__":
    main()
