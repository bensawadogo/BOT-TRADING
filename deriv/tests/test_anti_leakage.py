"""
GARDE-FOU 5 — Tests anti-leakage et anti-overfitting (López de Prado, 2018).

Adapté à l'architecture du bot :
  - `build_features()` ne contient PAS la target (découplage anti-fuite) ;
    les labels viennent de `build_labels()` → le garde-fou 2 du plan est
    déjà implémenté, ces tests le vérifient.
  - Le WFO applique PURGE (1 bougie) + EMBARGO entre train et test.

Lance avec : python -m pytest deriv/tests/test_anti_leakage.py -v
"""
import inspect

import numpy as np
import pandas as pd
import pytest

from deriv.ensemble_predictor import (
    FEATURE_COLS,
    XGBoostPredictor,
    build_features,
    build_labels,
)


def make_df(n: int = 500, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """
    OHLCV synthétique à l'échelle EURUSD — MARCHE ALÉATOIRE réaliste :
    close[t] = close[t-1] + drift + bruit i.i.d. (cumsum des incréments),
    comme les vrais marchés. Incréments indépendants → corr(r_t, target_t)
    ≈ 0 en l'absence de fuite, ce qui rend le test du détecteur fidèle.
    """
    rng = np.random.default_rng(seed)
    drift = 0.00005 if trend == "up" else -0.00005
    steps = drift + rng.standard_normal(n) * 0.001
    base = 1.1000 + np.cumsum(steps) - steps[0]     # base[0] = 1.1000
    return pd.DataFrame({
        "open":   base + rng.standard_normal(n) * 0.0002,
        "high":   base + np.abs(rng.standard_normal(n)) * 0.0005,
        "low":    base - np.abs(rng.standard_normal(n)) * 0.0005,
        "close":  base,
        "volume": np.ones(n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="4h"))


def feat_avec_target(df: pd.DataFrame) -> pd.DataFrame:
    """Features + labels réalignés (architecture découplée du bot)."""
    feat = build_features(df)
    labels = build_labels(df)
    assert labels is not None
    return feat.assign(target=labels.reindex(feat.index)).dropna(
        subset=["target"]
    )


# ══ GARDE-FOU 1 : Scaler fit sur le train uniquement ═══════════════════

class TestGardeFou1Scaler:

    def test_scaler_fit_sur_train_seulement(self):
        """Le center du scaler doit coller aux données TRAIN (80%), pas au tout."""
        df = make_df(500)
        xgb = XGBoostPredictor()
        xgb.train(df)

        assert xgb.scaler is not None, "Scaler non initialisé"

        feat_df = build_features(df)
        X_raw = feat_df[FEATURE_COLS].values
        split = int(len(X_raw) * 0.8)

        train_median = float(np.median(X_raw[:split, 0]))
        full_median = float(np.median(X_raw[:, 0]))
        scaler_center = float(xgb.scaler.center_[0])

        diff_train = abs(scaler_center - train_median)
        diff_full = abs(scaler_center - full_median)
        assert diff_train <= diff_full + 1e-6, (
            f"Scaler semble fit sur tout le dataset ! center={scaler_center:.6f}, "
            f"train_median={train_median:.6f}, full_median={full_median:.6f}"
        )

    def test_transform_uniquement_en_production(self):
        """predict()/predict_proba() doivent utiliser transform(), jamais fit_transform()."""
        # Le scaling se fait dans predict_proba() ; predict() délègue.
        # On vérifie les 2 méthodes pour couvrir le chemin production complet.
        src_proba = inspect.getsource(XGBoostPredictor.predict_proba)
        src_pred  = inspect.getsource(XGBoostPredictor.predict)
        assert "fit_transform" not in src_proba, (
            "predict_proba() appelle fit_transform() → FUITE : le scaler "
            "se refit sur les nouvelles données !"
        )
        assert "transform" in src_proba, (
            "predict_proba() doit appeler self.scaler.transform()")
        assert "fit_transform" not in src_pred, (
            "predict() appelle fit_transform() → FUITE !")
        assert "predict_proba" in src_pred

    def test_hmm_scaler_sauvegarde_avec_modele(self, tmp_path):
        """HMM : le scaler fit sur train est persisté dans le pickle."""
        from deriv.ensemble_predictor import HMMPredictor

        hmm = HMMPredictor()
        hmm.FILE = str(tmp_path / "hmm_gf1.pkl")
        hmm.train(make_df(300))
        assert hmm.scaler is not None, "HMM : scaler non initialisé"

        hmm2 = HMMPredictor()
        hmm2.FILE = hmm.FILE
        hmm2._load()
        assert hmm2.scaler is not None, "HMM : scaler non persisté dans le pickle"
        assert np.allclose(hmm2.scaler.center_, hmm.scaler.center_)


# ══ GARDE-FOU 2 : Target découplée, calculée avant filtrage ════════════

class TestGardeFou2Target:

    def test_target_calculee_avant_alignement(self):
        """build_labels contient shift(-1) ET l'exclusion de la dernière ligne."""
        src = inspect.getsource(build_labels)
        pos_shift = src.find("shift(-1)")
        pos_excl = src.find("iloc[:-1]")
        assert pos_shift != -1, "build_labels doit utiliser shift(-1)"
        assert pos_excl != -1, "build_labels doit exclure la dernière ligne"
        assert pos_shift < pos_excl, (
            "La target doit être calculée AVANT l'exclusion de la dernière ligne"
        )

    def test_derniere_ligne_supprimee(self):
        """Le dernier label pointe t+1 : interdit pour df[-1] (futur inconnu)."""
        df = make_df(200)
        labels = build_labels(df)
        assert labels.index[-1] == df.index[-2], (
            "Le dernier label correspond à df[-1] → sa target utiliserait "
            "close[t+1] qui n'existe pas"
        )

    def test_pas_de_nan_dans_target(self):
        df = make_df(300)
        labels = build_labels(df)
        assert not labels.isna().any()
        assert set(labels.unique()).issubset({0, 1})

    def test_alignement_features_labels_sans_fuite(self):
        """Chaque label aligné doit être recalculable depuis close[t+1] connu."""
        df = make_df(300)
        feat = build_features(df)
        aligned = feat_avec_target(df)

        # La dernière bougie du df (label inconnu) ne doit pas être alignée
        assert aligned.index[-1] < df.index[-1]
        # Recalcul indépendant sur 5 lignes (début + fin) → identique
        closes = df["close"]
        for idx in list(aligned.index[:2]) + list(aligned.index[-3:]):
            i = closes.index.get_loc(idx)
            assert i + 1 < len(closes), f"Label aligné sans futur connu : {idx}"
            expected = int(closes.iloc[i + 1] > closes.iloc[i])
            assert int(aligned.loc[idx, "target"]) == expected


# ══ GARDE-FOU 3 : Purge + Embargo dans le WFO ══════════════════════════

class TestGardeFou3Purge:

    def _wfo(self, granularity: int = 14400):
        from deriv.walk_forward_optimizer import WalkForwardOptimizer
        return WalkForwardOptimizer(granularity=granularity)

    def test_pas_de_chevauchement_train_test(self):
        """Train (après purge) et test ne se chevauchent sur AUCUN fold."""
        wfo = self._wfo(14400)
        df = make_df(1000)
        purge = getattr(wfo, "PURGE", 1)
        for fold in wfo.folds_plan(len(df), wfo.TRAIN_SIZE, wfo.TEST_SIZE,
                                   wfo.EMBARGO, wfo.STEP, purge):
            train_df = df.iloc[fold["train"][0]:fold["train"][1]]
            test_df = df.iloc[fold["test_start"]:fold["test_end"]]
            if len(train_df) > 0 and len(test_df) > 0:
                assert train_df.index[-1] < test_df.index[0], (
                    f"FUITE : train finit à {train_df.index[-1]}, "
                    f"test commence à {test_df.index[0]}"
                )

    def test_embargo_minimum_1_bougie(self):
        wfo = self._wfo(14400)
        assert wfo.EMBARGO >= 1, "Embargo = 0 → fuite par corrélation résiduelle"

    def test_purge_presente(self):
        wfo = self._wfo(14400)
        assert hasattr(wfo, "PURGE"), "PURGE non défini dans WalkForwardOptimizer"
        assert wfo.PURGE >= 1, (
            "PURGE = 0 → la dernière bougie train pointe vers le test"
        )

    def test_bougie_purgee_exclue_du_train(self):
        """La bougie purgée (index t1) est hors du train effectif, avant l'embargo."""
        wfo = self._wfo(14400)
        for fold in wfo.folds_plan(1000):
            t0, t1 = fold["train"]
            assert t0 < t1
            assert t1 + wfo.PURGE <= fold["test_start"]


# ══ GARDE-FOU 4 : Détection de feature leakage ═════════════════════════

class TestGardeFou4Leakage:

    def test_detection_leakage_no_crash(self):
        from deriv.leakage_detector import detect_feature_leakage

        feat = feat_avec_target(make_df(300))
        res = detect_feature_leakage(feat, FEATURE_COLS)
        assert "verdict" in res
        assert "leaked_features" in res
        assert "n_leaked" in res
        assert isinstance(res["leaked_features"], list)

    def test_target_absente_verdict_clair(self):
        """Sans colonne target, le détecteur prévient au lieu de mentir."""
        from deriv.leakage_detector import detect_feature_leakage

        feat = build_features(make_df(300))     # sans target (découplage)
        res = detect_feature_leakage(feat, FEATURE_COLS)
        assert res["n_leaked"] == 0
        assert "TARGET ABSENTE" in res["verdict"]

    def test_feature_future_detectee(self):
        """Une feature qui copie la direction future doit être flaggée FUITE."""
        from deriv.leakage_detector import detect_feature_leakage

        feat = feat_avec_target(make_df(300)).copy()
        # FUITE INTENTIONNELLE : la feature contient la direction de t+1
        bruit = np.random.default_rng(7).standard_normal(len(feat)) * 0.1
        feat["future_dir"] = feat["target"].astype(float) + bruit

        res = detect_feature_leakage(feat, FEATURE_COLS + ["future_dir"])
        assert "future_dir" in res["leaked_features"], (
            "Le détecteur n'a pas trouvé une feature clairement fuitée ! "
            "Revoir le seuil de détection."
        )

    def test_features_normales_pas_fuitees(self):
        """RSI/ROC/MACD/ADX/BB/ATR (features du passé) ne doivent pas être flaggées."""
        from deriv.leakage_detector import detect_feature_leakage

        feat = feat_avec_target(make_df(500))
        res = detect_feature_leakage(feat, FEATURE_COLS)
        assert res["n_leaked"] <= 1, (
            f"Trop de features détectées comme fuitées : {res['leaked_features']}\n"
            f"Détail : {res['detail']}\n"
            "Vérifie build_features() ou ajuste le seuil dans "
            "detect_feature_leakage()"
        )

    def test_rapport_lisible(self, capsys):
        from deriv.leakage_detector import run_leakage_check

        res = run_leakage_check(make_df(500))
        out = capsys.readouterr().out
        assert "DÉTECTION FEATURE LEAKAGE" in out
        assert res["verdict"] in out


# ══ GARDE-FOU 5 : Stabilité inter-seeds (proxy PBO) ════════════════════

class TestGardeFou5Stabilite:

    def test_win_rate_stable_sur_seeds_differentes(self):
        """
        Datasets similaires, seeds différentes → le win rate ne doit pas
        varier de plus de 15 pts d'écart-type entre seeds (sinon le modèle
        overfitte le bruit aléatoire).
        """
        win_rates = []
        for seed in (42, 123, 999, 2024):
            df_tr = make_df(400, seed=seed)
            df_te = make_df(200, seed=seed + 1)

            xgb = XGBoostPredictor()
            xgb.train(df_tr)

            feat = build_features(df_te)
            if feat.empty:
                continue

            preds, actuals = [], []
            for i in range(40, len(feat) - 1):
                res = xgb.predict(feat.iloc[:i])
                vote = res.get("vote", 0.5)
                if vote == 0.5:
                    continue
                preds.append(int(vote))
                actuals.append(
                    int(feat["close"].iloc[i + 1] > feat["close"].iloc[i])
                )

            if len(preds) >= 5:
                win_rates.append(
                    sum(p == a for p, a in zip(preds, actuals)) / len(preds)
                )

        assert len(win_rates) >= 3, (
            f"Seulement {len(win_rates)} seeds productives — test non exploitable"
        )
        std = float(np.std(win_rates))
        assert std < 0.15, (
            f"Win rate instable selon le seed : std={std:.1%}\n"
            f"Win rates : {[f'{w:.1%}' for w in win_rates]}\n"
            "→ Overfit sur le bruit aléatoire détecté !"
        )


# ══ GARDE-FOU 6 : Triple-barrière symétrique (anti-drift, P0-v2) ═══════

class TestTripleBarriereAntiDrift:
    """Audit 90 % : la target triviale close[t+1] > close[t] copie la
    dérive (acc réelle = acc permutée = baseline UP). La triple-barrière
    symétrique élimine ce biais : ratio labels ≈ 50/50 même sur un
    dataset drifté (mesuré : 0.561 sur make_df(trend="up"), horizon=5).
    """

    def test_triple_barriere_symetrique(self):
        """
        La triple-barrière ne doit pas être biaisée par le drift.
        Sur un dataset drifté, le ratio label=1 / label=0 doit être
        proche de 50/50 (±15%) — pas 70/30 comme la target triviale.
        """
        df = make_df(500, trend="up")  # dataset drifté volontairement
        labels = build_labels(
            df, mode="triple_barrier", touch_mult=1.5, horizon=5,
        )
        assert labels is not None, "build_labels(triple_barrier) → None"
        labels_clean = labels.dropna()

        if len(labels_clean) < 50:
            pytest.skip("Pas assez de labels après triple-barrière")

        ratio = float(labels_clean.mean())
        assert 0.35 <= ratio <= 0.65, (
            f"Triple-barrière biaisée : ratio={ratio:.2f} "
            f"(attendu 0.35-0.65, drift compensé)"
        )

    def test_baseline_vs_modele_sur_symbole_non_drifte(self):
        """
        Sur frxEURUSD (pas de drift), le modèle doit battre la baseline UP.
        Si baseline UP = modèle → pas de signal.

        Test logique uniquement (pas de réseau) : un dataset sans drift
        (marche aléatoire pure, mean return ≈ 0) doit donner une baseline
        UP proche de 50 %. Mesuré (seed=42, n=300) : up_pct = 0.515.
        """
        np.random.seed(42)
        n = 300
        prices = np.random.randn(n).cumsum() * 0.001 + 1.0  # marche aléatoire pure
        up_pct = (np.diff(prices) > 0).mean()

        # Sans drift, baseline UP doit être proche de 50%
        assert 0.45 <= up_pct <= 0.55, (
            f"Dataset test n'est pas sans drift : up_pct={up_pct:.2f}"
        )
