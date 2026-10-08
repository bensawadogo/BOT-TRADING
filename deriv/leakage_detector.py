"""
GARDE-FOU 4 — Détection automatique de feature leakage (López de Prado, 2018).

Principe : pour chaque feature, on compare la corrélation feature[t] → target[t]
avec la corrélation feature[t-1] → target[t]. Une feature qui contient de
l'information future (ex. built avec close[t+1]) corrèle beaucoup plus fort à
la target courante que sa version laggée → fuite détectée.

Usage :
    from deriv.leakage_detector import run_leakage_check
    run_leakage_check(df)          # rapport complet sur les features du bot

    from deriv.leakage_detector import detect_feature_leakage
    res = detect_feature_leakage(feat_df_avec_target, FEATURE_COLS)
"""
import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("LEAKAGE")


def detect_feature_leakage(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = "target",
    threshold_pvalue: float = 0.01,
) -> dict:
    """
    Pour chaque feature, teste si sa corrélation avec la target est
    significativement supérieure à la corrélation de sa version laggée (t-1).

    Si corr(feature[t], target[t]) >> corr(feature[t-1], target[t])
    → la feature contient de l'information future → FUITE.

    ⚠️ Le DataFrame doit contenir `target_col`. Avec l'architecture du bot
    (target découplée), injecter : df["target"] = build_labels(df).

    Retourne : {"leaked_features": [...], "n_leaked": int, "detail": {...},
                "verdict": str}
    """
    results: dict = {}

    if target_col not in df.columns:
        return {
            "leaked_features": [],
            "n_leaked": 0,
            "n_total": len(feature_cols),
            "detail": {},
            "verdict": ("⚠️ TARGET ABSENTE — vérification impossible "
                        "(injecte build_labels(df) avant l'appel)"),
        }

    target = df[target_col].astype(float)

    for col in feature_cols:
        if col not in df.columns:
            continue

        feat_t0 = df[col].astype(float)
        feat_lag1 = df[col].astype(float).shift(1)

        # Corrélation Pearson feature[t] → target[t]
        mask0 = feat_t0.notna() & target.notna()
        if mask0.sum() < 30 or float(np.std(feat_t0[mask0])) == 0 \
                or float(np.std(target[mask0])) == 0:
            # Feature constante ou échantillon trop faible → non testable
            results[col] = {"leaked": False, "pvalue": 1.0, "skipped": True}
            continue
        corr0, pval0 = stats.pearsonr(feat_t0[mask0], target[mask0])

        # Corrélation Pearson feature[t-1] → target[t]
        corr1, pval1 = 0.0, 1.0
        mask1 = feat_lag1.notna() & target.notna()
        if mask1.sum() >= 30 and float(np.std(feat_lag1[mask1])) > 0:
            corr1, pval1 = stats.pearsonr(feat_lag1[mask1], target[mask1])

        # Heuristique : corr actuelle >> corr laggée ET significative → fuite
        leaked = (abs(corr0) > abs(corr1) * 1.5) and (pval0 < threshold_pvalue)

        results[col] = {
            "leaked":    bool(leaked),
            "corr_t0":   round(float(corr0), 4),
            "corr_lag1": round(float(corr1), 4),
            "pvalue_t0": round(float(pval0), 6),
            "ratio":     round(abs(corr0) / max(abs(corr1), 1e-8), 2),
        }

    leaked_features = [k for k, v in results.items() if v.get("leaked")]

    return {
        "leaked_features": leaked_features,
        "n_leaked":        len(leaked_features),
        "n_total":         len(feature_cols),
        "detail":          results,
        "verdict":         "⚠️ FUITE DÉTECTÉE" if leaked_features
                           else "✅ Pas de fuite détectée",
    }


def run_leakage_check(df: pd.DataFrame) -> dict:
    """Lance la vérification sur les features réelles du bot et affiche le rapport."""
    from deriv.ensemble_predictor import build_features, build_labels, FEATURE_COLS

    feat_df = build_features(df)
    if feat_df.empty:
        print("Pas assez de données pour construire les features.")
        return {"leaked_features": [], "n_leaked": 0, "n_total": 0,
                "detail": {}, "verdict": "⚠️ Données insuffisantes"}

    # Target découplée (architecture anti-fuite) réalignée sur les features
    labels = build_labels(df)
    feat_df = feat_df.assign(target=labels.reindex(feat_df.index)).dropna(
        subset=["target"]
    )

    result = detect_feature_leakage(feat_df, FEATURE_COLS)

    print(f"\n{'='*50}")
    print("DÉTECTION FEATURE LEAKAGE")
    print(f"{'='*50}")
    print(f"Verdict : {result['verdict']}")
    print(f"Features testées : {result['n_total']}")
    print(f"Features suspectes : {result['n_leaked']}")

    if result["leaked_features"]:
        print("\n⚠️  Features à vérifier :")
        for f in result["leaked_features"]:
            d = result["detail"][f]
            print(f"  {f}: corr(t0)={d['corr_t0']:.3f} "
                  f"vs corr(lag)={d['corr_lag1']:.3f} "
                  f"(ratio={d['ratio']}x, p={d['pvalue_t0']})")
    else:
        print("\n✅ Toutes les features sont temporellement propres")

    print(f"{'='*50}\n")
    return result
