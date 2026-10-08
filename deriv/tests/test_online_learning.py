"""
════════════════════════════════════════════════════════════════════
TESTS V2 — Online Learning, Kelly, Session filter, Features V2
deriv/tests/test_online_learning.py
════════════════════════════════════════════════════════════════════
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np
import pandas as pd
import pytest

from deriv.online_learner import FEATURES_ONLINE, OnlineLearner


def make_df(n: int = 300) -> pd.DataFrame:
    """OHLCV synthétique pour tester build_features V2."""
    np.random.seed(42)
    base = 1000 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open":   base + np.random.randn(n) * 0.1,
        "high":   base + abs(np.random.randn(n)) * 0.5,
        "low":    base - abs(np.random.randn(n)) * 0.5,
        "close":  base,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })
    df.index = pd.date_range("2026-01-05 09:00", periods=n, freq="1h")
    return df


class TestOnlineLearner:
    @pytest.fixture
    def learner(self, tmp_path):
        return OnlineLearner(state_file=str(tmp_path / "online_test.pkl"))

    def test_import_river(self):
        # River doit être installé dans cet environnement (V2)
        try:
            import river  # noqa: F401
        except ImportError:
            pytest.skip("river non installé")

    def test_update_sans_crash(self, learner):
        r = learner.update({"hmm_confidence": 0.8, "rsi": 45.0}, label=1)
        assert r["ok"] is True
        assert r["n_samples"] == 1

    def test_predict_neutre_sans_historique(self, learner):
        r = learner.predict({"hmm_confidence": 0.8, "rsi": 45.0})
        assert r["vote"] in (0, 0.5, 1)
        assert r["active"] is False  # < 30 samples → inactif

    def test_features_manquantes_remplacees_par_zero(self, learner):
        """Un dict partiel ne doit jamais crasher (fill 0.0)."""
        r = learner.update({}, label=0)
        assert r["ok"] is True

    def test_apprend_un_pattern_separable(self, learner):
        """Sur un pattern linéairement séparable, le modèle doit prédire
        correctement APRÈS l'apprentissage (mesure via predict() pur,
        pas l'accuracy running pénalisée par le grace_period initial)."""
        for i in range(200):
            feat = {
                "hmm_confidence": 0.9 if i % 2 == 0 else 0.1,
                "rsi": 70.0 if i % 2 == 0 else 20.0,
            }
            learner.update(feat, label=int(i % 2 == 0))
        assert learner.n_samples == 200
        # Qualité de prédiction mesurée sur de nouveaux exemples sans update
        correct = 0
        for i in range(40):
            feat = {
                "hmm_confidence": 0.9 if i % 2 == 0 else 0.1,
                "rsi": 70.0 if i % 2 == 0 else 20.0,
            }
            r = learner.predict(feat)
            correct += int(r["vote"] == (1 if i % 2 == 0 else 0))
        assert correct / 40 > 0.55

    def test_persistance_state(self, learner, tmp_path):
        learner.update({"rsi": 50.0}, label=1)
        assert learner.n_samples == 1
        # Recharge un learner avec le même state_file → l'état est conservé
        l2 = OnlineLearner(state_file=learner.state_file)
        assert l2.n_samples == 1

    def test_predict_devient_actif_apres_30_samples(self, learner):
        for i in range(35):
            feat = {"hmm_confidence": 0.9 if i % 2 else 0.1}
            learner.update(feat, label=int(i % 2 == 0))
        r = learner.predict({"hmm_confidence": 0.9})
        assert r["active"] is True
        assert r["vote"] in (0, 0.5, 1)

    def test_features_online_liste_stable(self):
        assert "hmm_confidence" in FEATURES_ONLINE
        assert "session_overlap" in FEATURES_ONLINE


class TestKellyCriterion:
    def _rm(self, capital=100.0, auto_max=5.0):
        from deriv.risk_manager import RiskManager
        return RiskManager(capital=capital, auto_max=auto_max)

    def test_kelly_positif_borne_par_max_risk(self):
        rm = self._rm(capital=1000.0)
        # WR 70%, payout 0.85 : K = 0.70 - 0.30/0.85 = 0.347 → 25% = 8.7%
        # Borné par max_risk_pct (2%) → 0.02
        frac = rm.kelly_stake(0.70, 0.85)
        assert frac == pytest.approx(0.02, abs=1e-6)

    def test_kelly_negatif_si_esperance_perdante(self):
        rm = self._rm()
        # WR 50%, payout 0.85 : K = 0.50 - 0.50/0.85 = -0.088 → 0
        assert rm.kelly_stake(0.50, 0.85) == 0.0

    def test_kelly_25_pct_valeur_intermediaire(self):
        rm = self._rm()
        rm.max_risk_pct = 0.10  # relâche le plafond pour voir la formule
        # WR 60%, payout 0.85 : K = 0.60 - 0.40/0.85 = 0.129 → 25% = 3.2%
        frac = rm.kelly_stake(0.60, 0.85)
        assert frac == pytest.approx(0.1294 * 0.25, abs=1e-3)

    def test_get_optimal_stake_repli_sans_journal(self, tmp_path, monkeypatch):
        rm = self._rm(capital=1000.0)
        monkeypatch.setenv("JOURNAL_DB", str(tmp_path / "j.db"))
        stake, raison = rm.get_optimal_stake()
        assert stake > 0
        assert "insuffisant" in raison.lower()

    def test_get_optimal_stake_avec_wr_explicite(self):
        rm = self._rm(capital=1000.0, auto_max=50.0)
        rm.max_risk_pct = 0.10
        stake, raison = rm.get_optimal_stake(win_rate_running=0.60)
        # Kelly 25% = 3.24% de 1000 = 32.4$ → borné à 50 (auto_max)
        assert 0.35 <= stake <= 50.0
        assert "Kelly" in raison


class TestSessionFilter:
    def test_session_active_london(self):
        from deriv.ensemble_predictor import session_filter
        ok, raison = session_filter(pd.Timestamp("2026-01-05 09:00:00"))
        assert ok is True

    def test_session_active_overlap(self):
        from deriv.ensemble_predictor import session_filter
        ok, raison = session_filter(14)   # 14h UTC
        assert ok is True

    def test_session_asiatique_bloquee(self):
        from deriv.ensemble_predictor import session_filter
        ok, raison = session_filter(23)   # 23h UTC
        assert ok is False
        assert "hors session" in raison.lower()

    def test_session_heure_invalide_ne_crash_pas(self):
        from deriv.ensemble_predictor import session_filter
        ok, raison = session_filter("bidon")
        assert ok is True  # fallback : pas de filtre


class TestFeaturesV2:
    def test_feature_cols_v2_super_set_de_v1(self):
        from deriv.ensemble_predictor import FEATURE_COLS, FEATURE_COLS_V2
        assert set(FEATURE_COLS).issubset(set(FEATURE_COLS_V2))
        assert len(FEATURE_COLS_V2) > len(FEATURE_COLS)

    def test_nouvelles_colonnes_presentes(self):
        from deriv.ensemble_predictor import FEATURE_COLS_V2, build_features
        feat = build_features(make_df(300))
        for col in FEATURE_COLS_V2:
            assert col in feat.columns, f"colonne manquante : {col}"

    def test_pas_de_nan_apres_dropna_v2(self):
        from deriv.ensemble_predictor import FEATURE_COLS_V2, build_features
        feat = build_features(make_df(300))
        assert not feat[FEATURE_COLS_V2].isna().any().any()

    def test_sessions_coherentes(self):
        """À 14h UTC : london=1, newyork=1, overlap=1, asia=0."""
        from deriv.ensemble_predictor import build_features
        feat = build_features(make_df(300))
        row = feat.iloc[100]
        if row.name.hour == 14:
            assert row["london_active"] == 1
            assert row["newyork_active"] == 1
            assert row["overlap_active"] == 1
            assert row["asia_active"] == 0

    def test_volume_ratio_neutre_sans_volume(self):
        """Sans colonne volume → constantes neutres, pas de crash."""
        from deriv.ensemble_predictor import build_features
        df = make_df(200).drop(columns=["volume"])
        feat = build_features(df)
        assert (feat["volume_ratio_5"] == 1.0).all()
        assert (feat["volume_spike"] == 0.0).all()


class TestCPCV:
    def _stub_model(self):
        """Modèle trivial : prédit la classe majoritaire du train."""
        class StubModel:
            def fit(self, X, y):
                self.majority = int(np.bincount(y).argmax())
                return self

            def predict(self, X):
                return np.full(len(X), self.majority)
        return StubModel

    def test_structure_resultat(self):
        from deriv.cpcv_validator import CPCVValidator
        from deriv.ensemble_predictor import FEATURE_COLS, build_labels
        df = make_df(600)
        labels = build_labels(df)
        res = CPCVValidator(k=6, p=2, embargo=5).validate(
            df, self._stub_model(), FEATURE_COLS, labels=labels)
        assert res.get("erreur") is None
        assert res["n_paths"] > 0
        assert 0.0 <= res["wr_oos_moyen"] <= 1.0
        assert 0.0 <= res["pbo"] <= 1.0

    def test_embargo_pas_de_chevauchement(self):
        """Train et test ne partagent aucun indice (± embargo)."""
        from deriv.cpcv_validator import CPCVValidator
        cpcv = CPCVValidator(k=6, p=2, embargo=5)
        bounds = cpcv._fold_boundaries(600)
        test_ids = (0, 2)
        train_idx = set(cpcv._embargoed_train_idx(bounds, test_ids))
        for f in test_ids:
            s, e = bounds[f]
            for t in range(s, e):
                for i in range(t - 5, t + 6):
                    assert i not in train_idx

    def test_dataset_trop_court(self):
        from deriv.cpcv_validator import CPCVValidator
        from deriv.ensemble_predictor import FEATURE_COLS
        res = CPCVValidator(k=6, p=2).validate(
            make_df(100), self._stub_model(), FEATURE_COLS)
        assert "erreur" in res