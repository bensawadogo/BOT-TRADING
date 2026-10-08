"""
════════════════════════════════════════════════════════════════════
CPCV — Combinatorial Purged Cross-Validation
deriv/cpcv_validator.py
════════════════════════════════════════════════════════════════════

Démontré supérieur au WFO (Arian et al., 2024, Applied Mathematical
Finance) : PBO (Probability of Backtest Overfitting) plus faible.

Principe : au lieu d'un seul chemin train→test (WFO), le CPCV génère
C(k,p) chemins de backtest en combinant les folds, avec embargo entre
train et test. Chaque bougie est testée exactement p fois.

Métriques produites :
  - WR out-of-sample par chemin + moyenne/écart-type
  - PBO = fraction de chemins où le WR in-sample > out-of-sample
    (PBO < 0.4 = peu de surapprentissage ; > 0.6 = quasi-certain)

Le scaler est fit SUR LE TRAIN de chaque chemin uniquement
(garde-fou López de Prado conservé).
"""
from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class CPCVValidator:
    """CPCV simplifié avec embargo + mesure de PBO."""

    def __init__(self, k: int = 6, p: int = 2, embargo: int = 5):
        self.k = k          # nombre total de folds
        self.p = p          # folds en test par combinaison
        self.embargo = embargo  # bougies d'embargo autour du test

    def _fold_boundaries(self, n: int) -> list[tuple[int, int]]:
        """Divise n en k folds quasi égaux [(start, end), ...]."""
        fold_size = n // self.k
        bounds = []
        for i in range(self.k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.k - 1 else n
            bounds.append((start, end))
        return bounds

    def _embargoed_train_idx(self, bounds, test_fold_ids) -> list[int]:
        """Train = tous les folds sauf test, embargo de part et d'autre
        de chaque zone de test (exclut |i - t| <= embargo)."""
        test_set = set()
        for f in test_fold_ids:
            s, e = bounds[f]
            test_set.update(range(s, e))
        excluded = set()
        for t in test_set:
            excluded.update(range(max(0, t - self.embargo),
                                  t + self.embargo + 1))
        train_idx = []
        for fid in range(self.k):
            if fid in test_fold_ids:
                continue
            s, e = bounds[fid]
            train_idx.extend(
                i for i in range(s, e) if i not in excluded)
        return train_idx

    def validate(self, df: pd.DataFrame, model_class,
                 feature_cols: list[str], target_col: str = "target",
                 labels: pd.Series | None = None) -> dict:
        """
        Lance le CPCV.

        model_class : classe avec fit(X, y) / predict(X) (sklearn-like).
        labels      : Series alignée sur df.index ; si None, utilise
                      df[target_col] ou build_labels(df) (mode sign).
        """
        from deriv.ensemble_predictor import build_features, build_labels

        feat_df = build_features(df)
        if feat_df.empty:
            return {"erreur": "Features vides"}

        if labels is None:
            labels = (df[target_col] if target_col in df.columns
                      else build_labels(df))
        if labels is None:
            return {"erreur": "Labels indisponibles"}

        y_all = labels.reindex(feat_df.index).dropna()
        feat_df = feat_df.loc[y_all.index]
        X_raw = feat_df[feature_cols].values
        y_raw = y_all.values.astype(int)
        n = len(X_raw)
        if n < self.k * 30:
            return {"erreur": f"Dataset trop court ({n} lignes)"}

        bounds = self._fold_boundaries(n)
        test_combos = list(combinations(range(self.k), self.p))

        wr_oos_list: list[float] = []
        wr_is_list: list[float] = []
        errors = 0

        for test_fold_ids in test_combos:
            train_idx = self._embargoed_train_idx(bounds, test_fold_ids)
            test_idx = [i for f in test_fold_ids
                        for i in range(bounds[f][0], bounds[f][1])]
            if len(train_idx) < 50 or len(test_idx) < 10:
                continue
            try:
                scaler = RobustScaler()
                X_tr = scaler.fit_transform(X_raw[train_idx])
                X_te = scaler.transform(X_raw[test_idx])
                y_tr, y_te = y_raw[train_idx], y_raw[test_idx]

                model = model_class()
                model.fit(X_tr, y_tr)

                wr_is_list.append(float((model.predict(X_tr) == y_tr).mean()))
                wr_oos_list.append(float((model.predict(X_te) == y_te).mean()))
            except Exception as exc:
                errors += 1
                logger.warning("CPCV chemin %s échoué : %s",
                               test_fold_ids, exc)

        if not wr_oos_list:
            return {"erreur": "Aucun chemin valide", "n_errors": errors}

        n_overfit = sum(1 for a, b in zip(wr_is_list, wr_oos_list) if a > b)
        pbo = n_overfit / len(wr_oos_list)

        return {
            "n_paths": len(wr_oos_list),
            "n_errors": errors,
            "wr_oos_moyen": round(float(np.mean(wr_oos_list)), 4),
            "wr_oos_std": round(float(np.std(wr_oos_list)), 4),
            "wr_oos_min": round(float(np.min(wr_oos_list)), 4),
            "wr_oos_max": round(float(np.max(wr_oos_list)), 4),
            "wr_is_moyen": round(float(np.mean(wr_is_list)), 4),
            "pbo": round(pbo, 3),
            "interpretation": (
                "✅ PBO faible : peu de risque d'overfitting" if pbo < 0.4
                else "⚠️ PBO élevé : résultats potentiellement surestimés"
                if pbo < 0.6
                else "❌ PBO très élevé : overfitting quasi-certain"
            ),
            "profitable": bool(np.mean(wr_oos_list) >= 0.556),
        }
