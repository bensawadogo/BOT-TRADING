"""Test d'anti-fuite (permutation + baseline) pour CRASH500 M1."""
import sys, asyncio
sys.path.insert(0, '.')
import numpy as np
from deriv.data_collector import DerivDataCollector
from deriv.ensemble_predictor import build_features

async def main():
    c = DerivDataCollector('CRASH500', 60)
    await c.connect()
    df = await c.get_candle_history_full(target_count=1000)
    await c.close()
    feat = build_features(df).dropna()
    print(f'bougies_featurées : {len(feat)} | features : {feat.shape[1]}')

    # Target : close augmente dans les 5 prochaines bougies
    y = (feat['close'].shift(-5) > feat['close']).astype(int)
    y = y.reindex(feat.index).fillna(0).values[:len(feat)]
    # Features numériques (exclure close/open/high/low directement)
    exclude = ['close', 'open', 'high', 'low']
    X = feat.select_dtypes(include=[np.number]).drop(columns=exclude, errors='ignore').values[:len(feat)]
    print(f'shape_X={X.shape} shape_y={y.shape} balance_y={y.mean():.3f}')

    from sklearn.linear_model import LogisticRegression
    accs_real, accs_perm, accs_baseline = [], [], []
    for seed in range(5):
        np.random.seed(seed)
        idx = np.random.permutation(len(X))
        Xtr, Xte = X[idx[:700]], X[idx[700:]]
        ytr, yte = y[idx[:700]], y[idx[700:]]
        m = LogisticRegression(max_iter=200, random_state=42)
        m.fit(Xtr, ytr)
        accs_real.append(m.score(Xte, yte))
        # Permutation
        ytr_p = np.random.permutation(ytr)
        m2 = LogisticRegression(max_iter=200, random_state=42)
        m2.fit(Xtr, ytr_p)
        accs_perm.append(m2.score(Xte, yte))
        # Baseline : toujours UP
        accs_baseline.append((yte == 1).sum() / len(yte))

    print(f'acc_réelle : {np.mean(accs_real):.3f} ± {np.std(accs_real):.3f}')
    print(f'acc_permutée : {np.mean(accs_perm):.3f} ± {np.std(accs_perm):.3f}')
    print(f'acc_baseline_UP : {np.mean(accs_baseline):.3f}')
    print(f'rapport_signal/bruit : {np.mean(accs_real)/max(np.mean(accs_perm), 0.001):.2f}x')
    if np.mean(accs_perm) > 0.50:
        print('VERDICT : ⚠️ Fuite possible (perm > 50%) → investitgue feature alignment')
    elif np.mean(accs_real) > np.mean(accs_baseline) * 1.1:
        print('VERDICT : Signal réel plausible')
    else:
        print('VERDICT : Drift trivial (réelle ≈ baseline UP)')

asyncio.run(main())