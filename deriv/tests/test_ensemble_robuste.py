"""
═══════════════════════════════════════════════════════════════════
SUITE DE TESTS ROBUSTES — bot Deriv ensemble 4/5
deriv/tests/test_ensemble_robuste.py
═══════════════════════════════════════════════════════════════════

130+ tests : logique, edge cases, RÈGLE 4/5, walk-forward, statistiques
(Wilson, binomial, overfitting), risk manager, ré-entraînement hebdo.

Lance avec : python -m pytest deriv/tests/test_ensemble_robuste.py -v --tb=short
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deriv.ensemble_predictor import (
    EnsemblePredictor,
    FEATURE_COLS,
    HMMPredictor,
    KalmanMomentumPredictor,
    LSTMPredictor,
    MomentumPredictor,
    RSIDivergencePredictor,
    TrendStrengthPredictor,
    XGBoostPredictor,
    build_features,
)
from deriv.risk_manager import RiskManager, TradeResult

# Les 7 modèles de l'ensemble (ordre du vote)
MODELES = ["HMM", "XGBoost", "LSTM", "Kalman", "RSI",
           "TrendStrength", "Momentum"]
VOTES_TOUS_BUY = {m: 1 for m in MODELES}
VOTES_TOUS_SELL = {m: 0 for m in MODELES}
VOTES_TOUS_NEUTRE = {m: 0.5 for m in MODELES}


# ── FIXTURES ─────────────────────────────────────────────────────────

def make_df(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """
    DataFrame OHLCV synthétique avec tendance contrôlée.
    OHLC cohérent : high >= max(open, close), low <= min(open, close).
    """
    np.random.seed(42)
    t = np.arange(n)

    if trend == "up":
        base = 1000 + t * 0.5 + np.random.randn(n) * 2
    elif trend == "down":
        base = 1000 - t * 0.5 + np.random.randn(n) * 2
    elif trend == "flat":
        base = 1000 + np.random.randn(n) * 1
    elif trend == "volatile":
        base = 1000 + np.random.randn(n) * 20
    else:
        base = np.ones(n) * 1000

    open_ = base + np.random.randn(n) * 0.5
    close = base
    high = np.maximum(open_, close) + np.abs(np.random.randn(n)) * 1
    low = np.minimum(open_, close) - np.abs(np.random.randn(n)) * 1

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": np.random.randint(100, 1000, n).astype(float),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="1min")
    return df


def make_gbm_df(n: int = 300) -> pd.DataFrame:
    """Marché réaliste : mouvement brownien géométrique (prix log-normal)."""
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.005, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + rng.uniform(0.001, 0.004, size=n))
    low = close * (1.0 - rng.uniform(0.001, 0.004, size=n))
    open_ = low + rng.uniform(0.1, 0.9, size=n) * (high - low)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": rng.uniform(100, 500, size=n),
    }, index=pd.date_range("2025-01-01", periods=n, freq="5min"))
    return df


@pytest.fixture
def df_up():      return make_df(300, "up")
@pytest.fixture
def df_down():    return make_df(300, "down")
@pytest.fixture
def df_flat():    return make_df(300, "flat")
@pytest.fixture
def df_volatile(): return make_df(300, "volatile")
@pytest.fixture
def df_minimal(): return make_df(60, "up")
@pytest.fixture
def df_large():   return make_df(1000, "up")
@pytest.fixture
def df_gbm():     return make_gbm_df(300)


# ── TESTS FEATURE ENGINEERING ────────────────────────────────────────

class TestFeatureEngineering:

    def test_colonnes_presentes(self, df_up):
        feat = build_features(df_up)
        for col in FEATURE_COLS:
            assert col in feat.columns, f"Colonne manquante: {col}"

    def test_pas_de_nan(self, df_up, df_gbm):
        for feat in (build_features(df_up), build_features(df_gbm)):
            assert not feat[FEATURE_COLS].isna().any().any()

    def test_target_binaire(self, df_up):
        from deriv.ensemble_predictor import build_labels
        y = build_labels(df_up)
        assert y is not None
        assert set(y.dropna().unique()).issubset({0, 1})

    def test_target_sans_fuite_temporelle(self, df_up):
        """label[t] = (close[t+1] > close[t]) — jamais de t+2 ou plus."""
        from deriv.ensemble_predictor import build_labels
        y = build_labels(df_up)
        expected = (df_up["close"].shift(-1) > df_up["close"]).astype(int)
        # build_labels exclut la dernière ligne (label inconnu en live)
        assert set(y.index) == set(expected.iloc[:-1].index)
        for idx in y.index:
            assert y.loc[idx] == expected.loc[idx]

    def test_dataset_trop_petit(self):
        feat = build_features(make_df(5, "up"))
        assert len(feat) == 0 or feat.empty or len(feat) < 5

    def test_dataframe_vide_ne_crash_pas(self):
        feat = build_features(pd.DataFrame())
        assert isinstance(feat, pd.DataFrame)
        assert feat.empty

    def test_dataframe_none_ne_crash_pas(self):
        feat = build_features(None)
        assert isinstance(feat, pd.DataFrame)
        assert feat.empty

    def test_colonnes_manquantes_ne_crash_pas(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        feat = build_features(df)
        assert feat.empty

    def test_reproductibilite(self, df_up):
        pd.testing.assert_frame_equal(
            build_features(df_up.copy()), build_features(df_up.copy())
        )

    def test_volume_manquant_tolerant(self, df_up):
        df = df_up.drop(columns=["volume"])
        feat = build_features(df)
        assert not feat.empty   # volume par défaut = 1

    def test_index_temporel_conserves(self, df_up):
        feat = build_features(df_up)
        assert all(idx in df_up.index for idx in feat.index)


# ── TESTS MODÈLES INDIVIDUELS ────────────────────────────────────────

class TestHMMPredictor:

    def test_train_et_is_trained(self, df_large):
        hmm = HMMPredictor()
        hmm.train(df_large)
        assert hmm.is_trained
        assert hmm.model is not None

    def test_labels_trois_regimes(self, df_large):
        hmm = HMMPredictor()
        hmm.train(df_large)
        assert set(hmm.labels.values()) == {"Bull", "Bear", "Range"}

    def test_predict_structure(self, df_large):
        hmm = HMMPredictor()
        hmm.train(df_large)
        res = hmm.predict(df_large)
        for cle in ("regime", "viterbi", "confiance", "coherent", "vote"):
            assert cle in res, f"Clé manquante: {cle}"
        assert res["regime"] in ("Bull", "Bear", "Range")

    def test_confiance_entre_0_et_1(self, df_large):
        hmm = HMMPredictor()
        hmm.train(df_large)
        res = hmm.predict(df_large)
        assert 0.0 <= res["confiance"] <= 1.0

    def test_voter_valeurs_valides(self, df_large):
        """Votes HMM réels : {0, 0.3, 0.5, 0.7, 1} (0.7/0.3 = biais qualifié)."""
        hmm = HMMPredictor()
        hmm.train(df_large)
        res = hmm.predict(df_large)
        assert res["vote"] in (0, 0.3, 0.5, 0.7, 1)

    def test_donnees_insuffisantes_valueerror(self):
        """Moins de 60 bougies propres → ValueError (API réelle)."""
        hmm = HMMPredictor()
        with pytest.raises(ValueError):
            hmm.train(make_df(70, "up"))

    def test_sans_entrainement_neutre(self, df_up):
        hmm = HMMPredictor()
        hmm.model = None
        hmm.is_trained = False
        res = hmm.predict(df_up)
        assert res["vote"] == 0.5
        assert res["regime"] == "Range"

    def test_predict_dataframe_vide_neutre(self, df_large):
        """DataFrame vide → vote neutre sans crash (durcissement)."""
        hmm = HMMPredictor()
        hmm.train(df_large)
        res = hmm.predict(pd.DataFrame())
        assert res["vote"] == 0.5

    def test_serialisation_roundtrip(self, df_large, tmp_path):
        """Train → save → load → même régime prédit."""
        hmm1 = HMMPredictor()
        hmm1.FILE = str(tmp_path / "hmm_test.pkl")
        hmm1.train(df_large)
        pred1 = hmm1.predict(df_large)

        hmm2 = HMMPredictor()
        hmm2.FILE = hmm1.FILE
        hmm2._load()
        pred2 = hmm2.predict(df_large)

        assert pred1["regime"] == pred2["regime"]
        assert pred1["vote"] == pred2["vote"]


class TestXGBoostPredictor:

    def test_train_et_predict(self, df_large):
        xgb = XGBoostPredictor()
        xgb.train(df_large)
        assert xgb.is_trained
        res = xgb.predict(df_large)
        assert 0.0 <= res["proba_up"] <= 1.0
        assert res["vote"] in (0, 0.5, 1)

    def test_proba_extreme_vote_fort(self, df_large):
        """Cohérence vote↔proba↔seuils : vote 1 si p >= seuil_up de
        l'instance, 0 si p <= seuil_down, sinon neutre 0.5.

        P0-v2 : les seuils de l'instance sont NEUTRES (0.50/0.50) — la
        calibration 0.65/0.35 se fait GLOBALEMENT dans le WFO
        (`calibrate_global_threshold`). Avec des seuils neutres la
        décision limite est 0 ou 1 (le vote 0.5 n'apparaît qu'avec des
        seuils écartés). Depuis la target triple-barrière, p n'est plus
        tiré vers 1 par la dérive du dataset (avant : p > 0.65 garanti).
        """
        xgb = XGBoostPredictor()
        xgb.train(df_large)
        res = xgb.predict(df_large)
        p = res["proba_up"]
        eps = 1e-3   # p est arrondi à 3 décimales dans predict()
        if p >= xgb.threshold_up + eps:
            assert res["vote"] == 1, (p, xgb.threshold_up, res["vote"])
        elif p <= xgb.threshold_down - eps:
            assert res["vote"] == 0, (p, xgb.threshold_down, res["vote"])
        else:
            # Zone d'arrondi du seuil : décision limite (0 ou 1)
            assert res["vote"] in (0, 1), (p, res["vote"])

    def test_dataframe_vide_neutre(self, df_large):
        """DataFrame vide → neutre sans crash (durcissement edge case)."""
        xgb = XGBoostPredictor()
        xgb.train(df_large)
        res = xgb.predict(pd.DataFrame())
        assert res["vote"] == 0.5

    def test_petite_fenetre_neutre(self, df_large):
        """Fenêtre trop courte pour les features EMA50 → neutre."""
        xgb = XGBoostPredictor()
        xgb.train(df_large)
        res = xgb.predict(make_df(30, "up"))
        assert res["vote"] == 0.5

    def test_sans_entrainement_neutre(self):
        xgb = XGBoostPredictor()
        xgb.model = None
        xgb.is_trained = False
        res = xgb.predict(make_df(100))
        assert res["vote"] == 0.5
        assert res["proba_up"] == 0.5

    def test_train_donnees_insuffisantes(self):
        xgb = XGBoostPredictor()
        with pytest.raises(ValueError):
            xgb.train(make_df(50, "up"))

    def test_serialisation_roundtrip(self, df_large, tmp_path):
        xgb1 = XGBoostPredictor()
        xgb1.FILE = str(tmp_path / "xgb_test.pkl")
        xgb1.train(df_large)
        pred1 = xgb1.predict(df_large)

        xgb2 = XGBoostPredictor()
        xgb2.FILE = xgb1.FILE
        xgb2._load()
        pred2 = xgb2.predict(df_large)

        assert pred1["vote"] == pred2["vote"]

    def test_reproductibilite_random_state(self, df_large):
        """random_state=42 → deux entraînements prédisent pareil."""
        p1 = []
        for _ in range(2):
            x = XGBoostPredictor()
            x.train(df_large)
            p1.append(x.predict(df_large)["proba_up"])
        assert p1[0] == p1[1]


class TestLSTMFallback:

    def test_lstm_neutre_sans_tensorflow(self, df_up):
        """Sans TensorFlow (Python 3.14) → vote neutre 0.5, is_trained False."""
        lstm = LSTMPredictor()
        res = lstm.predict(df_up)
        assert res["vote"] == 0.5
        assert res["proba_up"] == 0.5

    def test_lstm_train_sans_tf_ne_crash_pas(self, df_large):
        """train() sans TF → warning + pas d'entraînement (dégradation propre)."""
        lstm = LSTMPredictor()
        try:
            import tensorflow  # noqa: F401
            pytest.skip("TensorFlow installé — fallback non applicable")
        except ImportError:
            pass
        lstm.train(df_large)
        assert not lstm.is_trained


class TestKalmanMomentum:

    def test_vote_buy_sur_tendance_forte(self):
        df = make_df(100, "up")
        df["close"] = 1000 + np.arange(100.0) * 0.5
        res = KalmanMomentumPredictor().predict(df)
        assert res["vote"] == 1

    def test_vote_sell_sur_tendance_baissiere(self):
        df = make_df(100, "down")
        df["close"] = 1000 - np.arange(100.0) * 0.5
        res = KalmanMomentumPredictor().predict(df)
        assert res["vote"] == 0

    def test_donnees_courtes_neutre(self):
        res = KalmanMomentumPredictor().predict(make_df(5, "up"))
        assert res["vote"] == 0.5

    def test_champs_requis(self, df_up):
        res = KalmanMomentumPredictor().predict(df_up)
        assert "trend" in res
        assert "mom" in res
        assert "vote" in res

    def test_vote_valeurs_valides(self, df_up):
        res = KalmanMomentumPredictor().predict(df_up)
        assert res["vote"] in (0, 0.5, 1)

    def test_trend_positive_sur_up(self, df_up):
        res = KalmanMomentumPredictor().predict(df_up)
        assert res["trend"] > 0


class TestRSIDivergence:

    def test_champs_requis(self, df_up):
        res = RSIDivergencePredictor().predict(df_up)
        for cle in ("rsi", "divergence_bull", "divergence_bear", "vote", "adx"):
            assert cle in res

    def test_rsi_entre_0_et_100(self, df_up, df_down):
        for df in (df_up, df_down):
            res = RSIDivergencePredictor().predict(df)
            assert 0 <= res["rsi"] <= 100

    def test_vote_entre_0_et_1(self, df_up, df_flat, df_volatile):
        for df in (df_up, df_flat, df_volatile):
            res = RSIDivergencePredictor().predict(df)
            assert 0.0 <= res["vote"] <= 1.0

    def test_donnees_courtes_neutre(self):
        res = RSIDivergencePredictor().predict(make_df(5, "up"))
        assert res["vote"] == 0.5
        assert res["divergence_bull"] is False
        assert res["divergence_bear"] is False

    def test_divergences_booleennes(self, df_gbm):
        res = RSIDivergencePredictor().predict(df_gbm)
        assert isinstance(res["divergence_bull"], bool)
        assert isinstance(res["divergence_bear"], bool)

    def test_ne_divergence_pas_simultanee(self, df_gbm):
        res = RSIDivergencePredictor().predict(df_gbm)
        assert not (res["divergence_bull"] and res["divergence_bear"])


class TestTrendStrength:

    def test_champs_requis(self, df_up):
        res = TrendStrengthPredictor().predict(df_up)
        for cle in ("adx", "ema_align", "confidence", "vote"):
            assert cle in res

    def test_donnees_courtes_neutre(self):
        res = TrendStrengthPredictor().predict(make_df(30, "up"))
        assert res["vote"] == 0.5
        assert res["ema_align"] == "NONE"

    def test_align_valeurs_valides(self, df_gbm):
        res = TrendStrengthPredictor().predict(df_gbm)
        assert res["ema_align"] in ("BULLISH", "BEARISH", "MIXED")

    def test_vote_entre_0_et_1(self, df_gbm, df_volatile):
        for df in (df_gbm, df_volatile):
            res = TrendStrengthPredictor().predict(df)
            assert 0.0 <= res["vote"] <= 1.0


class TestMomentum:

    def test_champs_requis(self, df_up):
        res = MomentumPredictor().predict(df_up)
        assert "roc" in res
        assert "vote" in res

    def test_donnees_courtes_neutre(self):
        res = MomentumPredictor().predict(make_df(10, "up"))
        assert res["vote"] == 0.5

    def test_vote_valeurs_valides(self, df_gbm):
        res = MomentumPredictor().predict(df_gbm)
        assert res["vote"] in (0, 0.5, 1)

    def test_bougies_neccessaires(self, df_up):
        """25 bougies minimum → calcul réel au-delà, neutre en dessous."""
        assert MomentumPredictor().predict(make_df(24, "up"))["vote"] == 0.5
        res = MomentumPredictor().predict(df_up)
        assert isinstance(res["roc"], float)


# ── TESTS ENSEMBLE — LA RÈGLE 4/5 (CŒUR DU BOT) ──────────────────────

def _stub_modele(vote: float) -> MagicMock:
    """Mock d'un modèle : toutes les clés lues par EnsemblePredictor.predict."""
    m = MagicMock()
    m.predict = MagicMock(return_value={
        "regime": "Bull", "viterbi": "Bull", "confiance": 0.8, "coherent": True,
        "proba_up": 0.75 if vote == 1 else (0.25 if vote == 0 else 0.5),
        "trend": 0.5, "mom": 0.2, "rsi": 35.0, "adx": 20.0,
        "divergence_bull": True, "divergence_bear": False,
        "ema_align": "MIXED", "roc": 0.0,
        "vote": vote,
    })
    return m


_ATTR_PAR_MODELE = {
    "HMM": "hmm", "XGBoost": "xgb", "LSTM": "lstm",
    "Kalman": "kalman", "RSI": "rsi",
    "TrendStrength": "trend", "Momentum": "momentum",
}


def _ensemble_avec_votes_fixes(votes: dict) -> dict:
    """
    EnsemblePredictor avec 7 modèles mockés qui retournent des votes
    prédéfinis → test pur de la logique de décision.
    """
    manquants = set(MODELES) - set(votes)
    assert not manquants, f"Votes manquants : {manquants}"

    e = EnsemblePredictor.__new__(EnsemblePredictor)   # pas de constructeur
    for nom in MODELES:
        setattr(e, _ATTR_PAR_MODELE[nom], _stub_modele(votes[nom]))
    e.MIN_VOTES_TO_TRADE = 4
    return e.predict(make_df(120))


class TestEnsembleRegle4sur5:
    """
    La RÈGLE ABSOLUE : au moins 4 votes pour trader, sinon HOLD.
    (Sur 7 modèles : >= 4 votes BUY ou >= 4 votes SELL. Jamais en dessous.)
    """

    @pytest.mark.parametrize("votes,expected_signal", [
        # 7/7 BUY → BUY
        (VOTES_TOUS_BUY, "BUY"),
        # 6/7 BUY → BUY
        ({**VOTES_TOUS_BUY, "RSI": 0.5}, "BUY"),
        # 5/7 BUY → BUY
        ({**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5}, "BUY"),
        # 4/7 BUY → BUY (le minimum verrouillé)
        ({**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}, "BUY"),
        # 3/7 BUY seulement → HOLD (règle 4/5 non atteinte)
        ({**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5,
          "Momentum": 0.5}, "HOLD"),
        # 3/7 SELL seulement → HOLD
        ({**VOTES_TOUS_SELL, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5,
          "Momentum": 0.5}, "HOLD"),
        # 7/7 SELL → SELL
        (VOTES_TOUS_SELL, "SELL"),
        # 4/7 SELL → SELL
        ({**VOTES_TOUS_SELL, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}, "SELL"),
        # Tous neutres → HOLD
        (VOTES_TOUS_NEUTRE, "HOLD"),
        # Conflit 3 BUY / 2 SELL / 2 neutres → HOLD
        ({**VOTES_TOUS_BUY, "HMM": 0, "XGBoost": 0,
          "TrendStrength": 0.5, "Momentum": 0.5}, "HOLD"),
        # 4 BUY + 3 SELL → BUY (priorité documentée du vote qualifié)
        ({**VOTES_TOUS_BUY, "RSI": 0, "Kalman": 0, "TrendStrength": 0}, "BUY"),
        # 4 SELL + 3 BUY → SELL (4 SELL atteint aussi le seuil)
        ({**VOTES_TOUS_SELL, "RSI": 1, "Kalman": 1, "TrendStrength": 1}, "SELL"),
    ])
    def test_regle_4_sur_5(self, votes, expected_signal):
        res = _ensemble_avec_votes_fixes(votes)
        assert res["signal"] == expected_signal, (
            f"Votes: {votes} → attendu: {expected_signal}, "
            f"obtenu: {res['signal']} "
            f"(buy:{res['buy_votes']} sell:{res['sell_votes']})"
        )

    def test_hold_ne_trade_jamais(self):
        votes = {**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5,
                 "TrendStrength": 0.5, "Momentum": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["signal"] == "HOLD"
        assert res["tradeable"] is False

    def test_buy_est_tradeable(self):
        votes = {**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["signal"] == "BUY"
        assert res["tradeable"] is True

    def test_sell_est_tradeable(self):
        votes = {**VOTES_TOUS_SELL, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["tradeable"] is True

    def test_compteurs_coherents_7_modeles(self):
        """buy_votes + sell_votes + hold_votes == 7 (7 modèles)."""
        votes = {**VOTES_TOUS_BUY, "RSI": 0, "Kalman": 0.5, "TrendStrength": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        total = (res["buy_votes"] + res["sell_votes"] + res["hold_votes"])
        assert total == 7

    def test_confiance_7_sur_7(self):
        res = _ensemble_avec_votes_fixes(VOTES_TOUS_BUY)
        assert res["confiance"] == 1.0

    def test_confiance_4_sur_7(self):
        votes = {**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["confiance"] == pytest.approx(4 / 7, abs=0.01)

    def test_confiance_5_sur_7(self):
        votes = {**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["confiance"] == pytest.approx(5 / 7, abs=0.01)

    def test_confiance_hold_zero(self):
        res = _ensemble_avec_votes_fixes(VOTES_TOUS_NEUTRE)
        assert res["confiance"] == 0.0

    def test_vote_0_7_compte_comme_buy(self):
        """Le HMM réel vote parfois 0.7 → compte comme vote BUY."""
        votes = {**VOTES_TOUS_NEUTRE, "HMM": 0.7, "XGBoost": 1, "LSTM": 1,
                 "Kalman": 1}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["buy_votes"] >= 4
        assert res["signal"] == "BUY"

    def test_vote_0_3_compte_comme_sell(self):
        votes = {**VOTES_TOUS_NEUTRE, "HMM": 0.3, "XGBoost": 0, "LSTM": 0,
                 "Kalman": 0}
        res = _ensemble_avec_votes_fixes(votes)
        assert res["sell_votes"] >= 4
        assert res["signal"] == "SELL"

    def test_min_votes_5_plus_strict(self):
        """Le seuil est configurable : 5 votes requis → 4 BUY ne suffit plus."""
        votes = {**VOTES_TOUS_BUY, "RSI": 0.5, "Kalman": 0.5, "TrendStrength": 0.5}
        res4 = _ensemble_avec_votes_fixes(votes)
        assert res4["signal"] == "BUY"
        # Avec MIN_VOTES = 5
        e = EnsemblePredictor.__new__(EnsemblePredictor)
        for nom in MODELES:
            setattr(e, _ATTR_PAR_MODELE[nom], _stub_modele(votes[nom]))
        e.MIN_VOTES_TO_TRADE = 5
        res5 = e.predict(make_df(120))
        assert res5["signal"] == "HOLD"

    def test_constante_verrouillee_4(self):
        """MIN_VOTES_TO_TRADE = 4 — NE JAMAIS BAISSER À 3."""
        assert EnsemblePredictor.MIN_VOTES_TO_TRADE == 4

    def test_detail_contient_7_modeles(self):
        res = _ensemble_avec_votes_fixes(VOTES_TOUS_BUY)
        assert set(res["detail"].keys()) == set(MODELES)

    def test_tous_les_modeles_interroges(self):
        """Chaque modèle est appelé exactement une fois par prédiction."""
        e = EnsemblePredictor.__new__(EnsemblePredictor)
        stubs = {nom: _stub_modele(1) for nom in MODELES}
        for nom in MODELES:
            setattr(e, _ATTR_PAR_MODELE[nom], stubs[nom])
        e.MIN_VOTES_TO_TRADE = 4
        e.predict(make_df(120))
        for nom, stub in stubs.items():
            stub.predict.assert_called_once()

    def test_votes_manquants_rejetes(self):
        """Le helper exige les 7 votes (protection contre les oublis)."""
        with pytest.raises(AssertionError):
            _ensemble_avec_votes_fixes({"HMM": 1, "XGBoost": 1})


# ── TESTS WALK-FORWARD ───────────────────────────────────────────────

class TestWalkForward:
    """
    Walk-Forward : entraîne sur N bougies, embargo, teste sur les M
    suivantes, répété sur toute la série → win rate réaliste.
    """

    # 1000 bougies, train 300 + embargo 30 + test 100, pas 130 → 5 folds
    PARAMS = dict(train_size=300, test_size=100, embargo=30, step=130)

    def _wfo_cls(self):
        from deriv.walk_forward_optimizer import WalkForwardOptimizer
        return WalkForwardOptimizer

    def test_plan_5_folds_minimum(self, df_large):
        """Mission : 5 folds minimum sur 1000 bougies."""
        plan = self._wfo_cls().folds_plan(1000, **self.PARAMS)
        assert len(plan) >= 5

    def test_plan_par_defaut_3000_bougies(self):
        plan = self._wfo_cls().folds_plan(3000)   # défauts 400/100/30/100
        assert len(plan) >= 5

    def test_embargo_entre_train_et_test(self, df_large):
        """test_start == train_end + PURGE + EMBARGO (purge + trou anti-fuite)."""
        wfo = self._wfo_cls()
        for f in wfo.folds_plan(1000, **self.PARAMS):
            t0, t1 = f["train"]          # train EFFECTIF (après purge)
            assert f["test_start"] == t1 + wfo.PURGE + 30
            assert f["purge"] >= 1

    def test_aucun_chevauchement_train_test(self, df_large):
        for f in self._wfo_cls().folds_plan(1000, **self.PARAMS):
            t0, t1 = f["train"]
            assert t1 <= f["test_start"] < f["test_end"] <= 1000
            assert t0 < t1

    def test_plan_serie_trop_courte(self):
        assert self._wfo_cls().folds_plan(100) == []

    def test_plan_reproductible(self):
        assert (self._wfo_cls().folds_plan(3000)
                == self._wfo_cls().folds_plan(3000))

    def test_ensemble_adapter_contrat_proba(self):
        """Non-régression : l'adapter expose TOUJOURS proba_up ∈ [0,1],
        directionnelle (≠0.5) quand le signal est BUY/SELL (sinon le WFO
        mesure un WR 0.0 biaisé — bug CRASH500 M1 5000 du 14/09).

        P0 (15/09) : échelle proba alignée sur _StatelessWrapper
        (BUY 0.8 / SELL 0.2) et HOLD qui n'expose AUCUNE proba (non tradé),
        afin que le seuil calibré sur le pool soit comparable entre modèles."""
        from deriv.walk_forward_optimizer import _EnsembleAdapter

        class _FakeEnsemble:
            def __init__(self, signal, confiance):
                self._signal, self._confiance = signal, confiance

            def predict(self, df):
                return {"signal": self._signal, "confiance": self._confiance}

        r_buy = _EnsembleAdapter(_FakeEnsemble("BUY", 0.71)).predict(None)
        assert set(("vote", "proba_up")) <= set(r_buy)
        assert r_buy["vote"] == 1.0
        # P0 (15/09) : échelle ALIGNÉE sur _StatelessWrapper (0.8 / 0.2).
        # La proba est ainsi comparable d'un modèle à l'autre dans le pool
        # out-of-sample ; `confiance` (échelle 4/7-7/7) désalignait
        # l'ensemble et faussait la calibration du seuil global.
        assert r_buy["proba_up"] == _EnsembleAdapter.PROBA_BUY

        r_sell = _EnsembleAdapter(_FakeEnsemble("SELL", 0.57)).predict(None)
        assert r_sell["vote"] == 0.0
        assert 0.0 <= r_sell["proba_up"] <= 1.0
        assert r_sell["proba_up"] != 0.5

        r_hold = _EnsembleAdapter(_FakeEnsemble("HOLD", 0.0)).predict(None)
        # P0 (15/09) : HOLD n'est PAS un trade → AUCUNE proba exposée. Un
        # 0.5 constant serait élu « UP » par tout seuil ≤ 0.50 et le win
        # rate de l'ensemble redeviendrait le taux de base de l'indice.
        assert r_hold["vote"] == 0.5
        assert r_hold["proba_up"] is None
        assert r_hold["p_useful"] is None

        # Signal inconnu / vide → traité comme neutre (non tradé).
        r_unknown = _EnsembleAdapter(_FakeEnsemble("INCONNU", 0.9)).predict(None)
        assert r_unknown["proba_up"] is None

    def test_run_produit_folds_et_wr_coherent(self, df_large):
        """WFO réel (XGBoost) : >= 5 folds, win rate ∈ [0, 1], trades > 0."""
        from deriv.walk_forward_optimizer import WalkForwardOptimizer

        wfo = WalkForwardOptimizer(**self.PARAMS, eval_ensemble=False,
                                   eval_in_sample=False)
        wfo.stateless_models = {}          # XGBoost seul (rapidité des tests)
        res = wfo.run(df_large)

        assert "XGBoost" in res, "XGBoost n'a produit aucun fold"
        stats = res["XGBoost"]
        assert stats["n_folds"] >= 5
        assert 0.0 <= stats["win_rate_moyen"] <= 1.0
        assert stats["n_trades_total"] > 0
        assert len(stats["detail_folds"]) == stats["n_folds"]

    def test_run_donnees_insuffisantes(self):
        with pytest.raises(ValueError):
            self._wfo_cls()().run(make_df(300, "up"))

    def test_win_rate_logge_et_lisible(self, df_large, capsys):
        from deriv.walk_forward_optimizer import WalkForwardOptimizer

        wfo = WalkForwardOptimizer(**self.PARAMS, eval_ensemble=False,
                                   eval_in_sample=False)
        wfo.stateless_models = {}
        res = wfo.run(df_large)
        wfo.print_report(res)
        out = capsys.readouterr().out
        assert "WALK-FORWARD" in out
        assert "XGBoost" in out

    def test_wr_significatif_dans_plan(self, df_large):
        """Le WR pondéré par fold doit rester dans [0, 1] par construction."""
        from deriv.walk_forward_optimizer import WalkForwardOptimizer

        wfo = WalkForwardOptimizer(**self.PARAMS, eval_ensemble=False,
                                   eval_in_sample=False)
        wfo.stateless_models = {}
        res = wfo.run(df_large)
        for fold in res["XGBoost"]["detail_folds"]:
            assert 0.0 <= fold["win_rate"] <= 1.0
            assert fold["n_trades"] > 0


# ── TESTS STATISTIQUES DE ROBUSTESSE ─────────────────────────────────

class TestStatistiques:
    """Wilson, p-value binomiale, détection d'overfitting — sans entraînement."""

    def _wfo(self):
        from deriv.walk_forward_optimizer import WalkForwardOptimizer
        return WalkForwardOptimizer.__new__(WalkForwardOptimizer)

    # ── Intervalle de Wilson ─────────────────────────────────────────
    def test_wilson_0_trades(self):
        from deriv.walk_forward_optimizer import wilson_interval
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_wilson_100_pourcent_narrow(self):
        from deriv.walk_forward_optimizer import wilson_interval
        lo, hi = wilson_interval(100, 100)
        assert lo > 0.95
        assert hi <= 1.0

    def test_wilson_contient_le_wr(self):
        from deriv.walk_forward_optimizer import wilson_interval
        lo, hi = wilson_interval(60, 100)
        assert lo < 0.6 < hi

    def test_wilson_bornes_valides(self):
        from deriv.walk_forward_optimizer import wilson_interval
        for wins in (0, 10, 25, 50, 90):
            lo, hi = wilson_interval(wins, 100)
            assert 0.0 <= lo <= hi <= 1.0

    def test_wilson_se_retrécit_avec_n(self):
        from deriv.walk_forward_optimizer import wilson_interval
        lo20, hi20 = wilson_interval(11, 20)
        lo500, hi500 = wilson_interval(275, 500)
        assert (hi20 - lo20) > (hi500 - lo500)

    def test_wilson_hasard_ne_rejette_pas_0_5(self):
        from deriv.walk_forward_optimizer import wilson_interval
        lo, hi = wilson_interval(50, 100)
        assert lo < 0.5 < hi   # un WR de hasard n'est jamais "significatif"

    # ── P-value binomiale ────────────────────────────────────────────
    def test_binomial_parfait_significatif(self):
        from deriv.walk_forward_optimizer import binomial_pvalue
        assert binomial_pvalue(30, 30) < 0.001

    def test_binomial_hasard_non_significatif(self):
        from deriv.walk_forward_optimizer import binomial_pvalue
        assert binomial_pvalue(15, 30) > 0.05

    def test_binomial_0_trades(self):
        from deriv.walk_forward_optimizer import binomial_pvalue
        assert binomial_pvalue(0, 0) == 1.0

    def test_binomial_monotone_avec_wins(self):
        from deriv.walk_forward_optimizer import binomial_pvalue
        assert binomial_pvalue(65, 100) < binomial_pvalue(55, 100)

    # ── Détection d'overfitting ──────────────────────────────────────
    def test_overfitting_ecart_in_out(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        diag = diagnostic_overfitting(0.90, 0.40, [0.40, 0.41, 0.39], 300)
        assert diag["overfitting"] is True
        assert any("surapprentissage" in s for s in diag["signaux"])

    def test_pas_d_overfitting_si_concordant(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        diag = diagnostic_overfitting(0.58, 0.57, [0.56, 0.57, 0.58], 300)
        assert diag["overfitting"] is False
        assert diag["fiabilite"] == "BONNE"

    def test_overfitting_instabilite_folds(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        diag = diagnostic_overfitting(0.60, 0.58, [0.2, 0.9, 0.3, 0.85], 300)
        assert diag["overfitting"] is True
        assert any("instable" in s for s in diag["signaux"])

    def test_overfitting_seuil_std_20_pourcent(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        folds = [0.40 if i % 2 == 0 else 0.78 for i in range(45)]
        diag = diagnostic_overfitting(0.58, 0.58, folds, 300)
        assert 0.18 < diag["ecart_type_folds"] < 0.20
        assert diag["overfitting"] is False

    def test_overfitting_echantillon_faible(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        diag = diagnostic_overfitting(None, 0.60, [0.60, 0.61], 5)
        assert diag["overfitting"] is True
        assert any("échantillon" in s for s in diag["signaux"])

    def test_overfitting_sans_in_sample(self):
        from deriv.walk_forward_optimizer import diagnostic_overfitting
        diag = diagnostic_overfitting(None, 0.60, [0.60, 0.61], 300)
        assert diag["ecart_in_out"] == 0.0

    # ── Verdict démo via _summarize ──────────────────────────────────
    def test_demo_ready_tous_criteres(self):
        folds = [{"fold": i, "n_trades": 60, "win_rate": 0.60}
                 for i in range(1, 6)]
        s = self._wfo()._summarize({"X": folds}, 5)["X"]
        assert s["profitable"] is True
        assert s["demo_ready"] is True

    def test_demo_ready_refuse_wr_sous_seuil(self):
        folds = [{"fold": i, "n_trades": 60, "win_rate": 0.50}
                 for i in range(1, 6)]
        s = self._wfo()._summarize({"X": folds}, 5)["X"]
        assert s["demo_ready"] is False
        assert s["profitable"] is False

    def test_demo_ready_refuse_trop_peu_de_trades(self):
        folds = [{"fold": i, "n_trades": 20, "win_rate": 0.70}
                 for i in range(1, 6)]
        s = self._wfo()._summarize({"X": folds}, 5)["X"]
        assert s["profitable"] is True
        assert s["demo_ready"] is False   # 100 trades < 200

    def test_summary_champs_statistiques_presents(self):
        folds = [{"fold": 1, "n_trades": 50, "win_rate": 0.6},
                 {"fold": 2, "n_trades": 50, "win_rate": 0.7}]
        s = self._wfo()._summarize({"X": folds}, 2)["X"]
        for cle in ("win_rate_moyen", "win_rate_std", "wilson_95",
                    "p_value_vs_hasard", "meilleur_que_hasard",
                    "diagnostic_overfitting"):
            assert cle in s
        lo, hi = s["wilson_95"]
        assert lo <= s["win_rate_moyen"] <= hi


# ── TESTS RISK MANAGER ───────────────────────────────────────────────

class TestRiskManager:
    """Gestion du capital : stops, cooldown, breakeven, confirmation."""

    def _rm(self, **kw) -> RiskManager:
        rm = RiskManager(capital=100.0, **kw)
        # Valeurs explicites (déterministes, indépendantes du .env)
        rm.max_risk_pct = 0.02
        rm.daily_stop_pct = 0.05
        rm.max_consecutive = 3
        return rm

    def test_max_stake_2pct(self):
        assert self._rm().max_stake() == pytest.approx(2.0, abs=0.01)

    def test_bloque_si_stop_journalier(self):
        rm = self._rm()
        rm.daily_loss = 5.1   # -5.1$ sur capital 100, stop à 5%
        ok, raison = rm.check(2.0, 80.0)
        assert not ok
        assert "journalière" in raison.lower()

    def test_bloque_apres_3_pertes(self):
        rm = self._rm()
        for _ in range(3):
            rm.record(TradeResult(symbol="R_75", pnl=-0.5))
        ok, raison = rm.check(2.0, 80.0)
        assert not ok
        assert "3" in raison or "consec" in raison.lower()

    def test_remet_compteur_apres_gain(self):
        rm = self._rm()
        rm.record(TradeResult("R_75", -2.0))
        rm.record(TradeResult("R_75", -2.0))
        rm.record(TradeResult("R_75", +5.0))   # gain → reset
        assert rm.consecutive_loss == 0
        assert rm.cooldown_until is None

    def test_bloque_montant_depasse_max(self):
        ok, raison = self._rm().check(3.0, 80.0)   # max = 2.0
        assert not ok
        assert "CONFIRMATION_REQUISE" in raison

    def test_bloque_confiance_insuffisante(self):
        """Verrou breakeven : confiance < 55.6% → refusé statistiquement."""
        ok, raison = self._rm().check(1.0, 50.0)
        assert not ok
        assert "breakeven" in raison.lower()

    def test_ok_si_confiance_au_dessus_breakeven(self):
        ok, _ = self._rm().check(1.0, 57.2)   # signal 4/7 = 57.1%
        assert ok

    def test_ok_si_tout_valide(self):
        ok, raison = self._rm().check(2.0, 80.0)
        assert ok
        assert raison == "OK"

    def test_confirmation_si_montant_trop_eleve(self):
        rm = self._rm()
        rm.auto_max = 5.0
        ok, raison = rm.check(6.0, 80.0)
        assert not ok
        assert "CONFIRMATION_REQUISE" in raison

    def test_winrate_correct(self):
        rm = self._rm()
        rm.record(TradeResult("R_75", +2.0))
        rm.record(TradeResult("R_75", -2.0))
        rm.record(TradeResult("R_75", +2.0))
        assert rm.winrate == pytest.approx(2 / 3, abs=0.01)

    def test_winrate_zero_sans_trades(self):
        assert self._rm().winrate == 0.0

    def test_cooldown_actif_bloque(self):
        from deriv.risk_manager import datetime as _dt

        rm = self._rm()
        rm.cooldown_until = _dt.now() + timedelta(minutes=10)
        ok, raison = rm.can_trade()
        assert not ok
        assert "refroidissement" in raison.lower()

    def test_capital_actuel_diminue_avec_pertes(self):
        rm = self._rm()
        rm.record(TradeResult("R_75", -3.0))
        assert rm.capital_actuel == 97.0

    def test_record_float_equivalent_traderesult(self):
        rm1, rm2 = self._rm(), self._rm()
        rm1.record(-2.0)
        rm2.record(TradeResult("R_75", -2.0))
        assert rm1.daily_loss == rm2.daily_loss == 2.0
        assert rm1.consecutive_loss == rm2.consecutive_loss == 1

    def test_stats_cles_pour_telegram(self):
        stats = self._rm().stats
        for cle in ("winrate_jour", "total_trades", "daily_loss",
                    "consecutive_loss", "capital_actuel", "in_cooldown"):
            assert cle in stats

    def test_decide_auto_ok(self):
        d = self._rm().decide(1.0)
        assert d.allowed and d.mode == "auto"

    def test_decide_manuel_bloque_sans_confirmation(self):
        d = self._rm().decide(50.0, confirmed=False)
        assert not d.allowed and d.mode == "manual"

    def test_decide_manuel_ok_avec_confirmation(self):
        d = self._rm().decide(50.0, confirmed=True)
        assert d.allowed and d.mode == "manual"

    def test_traderesult_champs_optionnels(self):
        tr = TradeResult(symbol="R_75", pnl=1.0)
        assert tr.contract_type is None
        assert tr.stake is None


# ── TESTS COLLECTEUR DE DONNÉES ──────────────────────────────────────

class TestDerivDataCollector:

    def test_agregation_ticks_en_bougies(self):
        from deriv.data_collector import DerivDataCollector

        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 3601):   # 3600 ticks → ~60 bougies
            col.process_tick(epoch, 100.0 + 0.001 * epoch)
        assert len(col.candles) >= 60

    def test_ohlc_coherent(self):
        from deriv.data_collector import DerivDataCollector

        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 3601):
            col.process_tick(epoch, 100.0 + np.sin(epoch / 50.0))
        for c in col.candles:
            assert c["low"] <= c["open"] <= c["high"]
            assert c["low"] <= c["close"] <= c["high"]

    def test_vide_avant_50_bougies(self):
        from deriv.data_collector import DerivDataCollector

        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 200):
            col.process_tick(epoch, 100.0)
        assert col.get_dataframe().empty

    def test_volume_nombre_de_ticks(self):
        from deriv.data_collector import DerivDataCollector

        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 3601):
            col.process_tick(epoch, 100.0)
        # 60 ticks par bougie de 60s (1 tick/s) — 1ère bougie partielle exclue
        assert all(c["volume"] >= 2 for c in col.candles)
        assert col.candles[-1]["volume"] == 60

    def test_get_dataframe_colonnes_et_index(self):
        from deriv.data_collector import DerivDataCollector

        col = DerivDataCollector("frxEURUSD", tick_size=60)
        for epoch in range(1, 3601):
            col.process_tick(epoch, 100.0)
        df = col.get_dataframe()
        assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_symboles_definis(self):
        from deriv.data_collector import DerivDataCollector

        for sym in ("frxEURUSD", "R_75", "R_50", "cryBTCUSD"):
            assert sym in DerivDataCollector.SYMBOLES

    def test_tick_size_minimum_1(self):
        from deriv.data_collector import DerivDataCollector

        assert DerivDataCollector("frxEURUSD", tick_size=0).tick_size == 1


# ── TESTS SYMBOLE × TIMEFRAME ────────────────────────────────────────

class _FakeCollector:
    def __init__(self, symbol, tick_size=60, connection=None):
        self.symbol = symbol

    async def connect(self): return self

    async def close(self): pass

    async def get_candle_history(self, count):
        return make_df(700, "up")

    async def get_candle_history_full(self, target_count):
        return make_df(700, "up")


class _FakeCollectorCourte(_FakeCollector):
    async def get_candle_history(self, count):
        return make_df(100, "up")

    async def get_candle_history_full(self, target_count):
        return make_df(100, "up")


class _FakeCollectorBoom(_FakeCollector):
    async def connect(self):
        raise ConnectionError("réseau indisponible")


class _FakeWFO:
    def __init__(self, granularity: int = 60):
        # Signaturé comme le vrai : WFO adaptatif par granularité (H1/H4)
        self.granularity = granularity

    def run(self, df):
        return {"XGBoost": {"win_rate_moyen": 0.62, "win_rate_std": 0.05,
                            "n_folds": 5, "n_trades_total": 120,
                            "detail_folds": []}}


class TestSymbolOptimizer:

    def test_symboles_12_combinaisons(self):
        """12 combos : vrais marchés H4/H1 + métaux + synthétiques H4 only."""
        from deriv.symbol_optimizer import SYMBOLES_A_TESTER

        assert len(SYMBOLES_A_TESTER) == 12
        assert len({(s, g) for s, g, _ in SYMBOLES_A_TESTER}) == 12
        # Vrais marchés forex bien présents (patterns humains, pas RNG)
        for sym, gran in [("frxEURUSD", 14400), ("frxGBPUSD", 14400),
                          ("frxUSDJPY", 14400), ("frxEURGBP", 14400),
                          ("frxEURUSD", 3600), ("frxGBPUSD", 3600)]:
            assert (sym, gran) in {(s, g) for s, g, _ in SYMBOLES_A_TESTER}, \
                f"Combo manquant : {sym} {gran}"
        # Métaux (Or = trending, bon pour HMM)
        assert ("frxXAUUSD", 14400) in {(s, g) for s, g, _ in SYMBOLES_A_TESTER}
        # Synthétiques uniquement en H4 (M1 = bruit pur, déjà établi)
        synth = [(s, g) for s, g, _ in SYMBOLES_A_TESTER
                 if s in ("R_75", "R_50")]
        assert all(g >= 14400 for _, g in synth), \
            "Les synthétiques ne doivent être testés qu'en H4+"

    def test_count_adapte_selon_granularite(self):
        """H4→500, H1→1000, M5/M15→2000, M1→3000."""
        from deriv.symbol_optimizer import _count_pour_granularity

        assert _count_pour_granularity(14400) == 500
        assert _count_pour_granularity(3600) == 1000
        assert _count_pour_granularity(300) == 2000
        assert _count_pour_granularity(60) == 3000

    def test_classement_tri_par_wr_desc(self):
        from deriv.symbol_optimizer import classer_resultats

        res = classer_resultats([
            {"label": "a", "win_rate": 0.5},
            {"label": "b", "win_rate": 0.7},
            {"label": "c", "win_rate": 0.6},
        ])
        assert [r["label"] for r in res] == ["b", "c", "a"]

    def test_classement_erreurs_en_dernier(self):
        from deriv.symbol_optimizer import classer_resultats

        res = classer_resultats([
            {"label": "err", "win_rate": 0, "erreur": "x"},
            {"label": "ok", "win_rate": 0.56},
        ])
        assert res[-1]["label"] == "err"
        assert res[0]["label"] == "ok"

    def test_classement_stabilite_tiebreaker(self):
        from deriv.symbol_optimizer import classer_resultats

        res = classer_resultats([
            {"label": "volatil", "win_rate": 0.6, "win_rate_std": 0.20},
            {"label": "stable", "win_rate": 0.6, "win_rate_std": 0.03},
        ])
        assert res[0]["label"] == "stable"

    def test_tester_symbole_ok(self, monkeypatch):
        from deriv import data_collector, symbol_optimizer, walk_forward_optimizer

        monkeypatch.setattr(data_collector, "DerivDataCollector", _FakeCollector)
        monkeypatch.setattr(walk_forward_optimizer,
                            "WalkForwardOptimizer", _FakeWFO)
        res = asyncio.run(symbol_optimizer.tester_symbole("R_75", 60, "Test R_75"))
        assert res["win_rate"] == pytest.approx(0.62, abs=0.001)
        assert res["profitable"] is True
        assert res["n_folds"] == 5
        assert res["meilleur_modele"] == "XGBoost"

    def test_tester_symbole_pas_assez_de_donnees(self, monkeypatch):
        from deriv import data_collector, symbol_optimizer, walk_forward_optimizer

        monkeypatch.setattr(data_collector, "DerivDataCollector",
                            _FakeCollectorCourte)
        monkeypatch.setattr(walk_forward_optimizer,
                            "WalkForwardOptimizer", _FakeWFO)
        res = asyncio.run(symbol_optimizer.tester_symbole("R_75", 60, "Test court"))
        assert "erreur" in res

    def test_tester_symbole_exception_capturee(self, monkeypatch):
        from deriv import data_collector, symbol_optimizer, walk_forward_optimizer

        monkeypatch.setattr(data_collector, "DerivDataCollector",
                            _FakeCollectorBoom)
        monkeypatch.setattr(walk_forward_optimizer,
                            "WalkForwardOptimizer", _FakeWFO)
        res = asyncio.run(symbol_optimizer.tester_symbole("R_75", 60, "Test erreur"))
        assert "erreur" in res
        assert res["win_rate"] == 0


# ── TESTS RÉ-ENTRAÎNEMENT HEBDOMADAIRE ───────────────────────────────

class _FakeEnsemble:
    instances = []

    def __init__(self):
        self.trained = None
        _FakeEnsemble.instances.append(self)

    def train_all(self, df):
        self.trained = df


class TestWeeklyRetrain:

    def _sched(self, tmp_path, **kw):
        from deriv.weekly_retrain import RetrainScheduler

        kw.setdefault("state_file", str(tmp_path / "state.json"))
        return RetrainScheduler(**kw)

    def test_pas_d_etat_retrain_requis(self, tmp_path):
        assert self._sched(tmp_path).needs_retrain() is True

    def test_vient_d_etre_entreine_pas_de_retrain(self, tmp_path):
        s = self._sched(tmp_path)
        s.mark_trained()
        assert s.needs_retrain() is False

    def test_huit_jours_retrain_requis(self, tmp_path):
        s = self._sched(tmp_path)
        s.mark_trained(now=datetime.now() - timedelta(days=8))
        assert s.needs_retrain() is True

    def test_six_jours_pas_de_retrain(self, tmp_path):
        s = self._sched(tmp_path)
        s.mark_trained(now=datetime.now() - timedelta(days=6))
        assert s.needs_retrain() is False

    def test_intervalle_personnalisable(self, tmp_path):
        s = self._sched(tmp_path, interval_days=1.0)
        s.mark_trained(now=datetime.now() - timedelta(days=2))
        assert s.needs_retrain() is True

    def test_etat_corrrompu_retrain_requis(self, tmp_path):
        p = tmp_path / "state.json"
        p.write_text("{ invalide", encoding="utf-8")
        s = self._sched(tmp_path)
        s.state_file = str(p)
        assert s.needs_retrain() is True

    def test_roundtrip_etat_persiste(self, tmp_path):
        s1 = self._sched(tmp_path)
        s1.mark_trained(meta={"symbol": "R_75"})
        s2 = self._sched(tmp_path)          # nouvelle instance, même fichier
        assert s2.needs_retrain() is False
        assert s2.history()[0]["symbol"] == "R_75"

    def test_historique_borne_a_50(self, tmp_path):
        s = self._sched(tmp_path)
        for i in range(60):
            s.mark_trained()
        assert len(s.history()) == 50

    def test_retrain_avec_df_fourni(self, tmp_path, monkeypatch):
        from deriv import ensemble_predictor, weekly_retrain

        _FakeEnsemble.instances = []
        monkeypatch.setattr(ensemble_predictor, "EnsemblePredictor",
                            _FakeEnsemble)
        s = self._sched(tmp_path, symbol="R_75", granularity=60)
        rep = asyncio.run(s.retrain_async(df=make_df(400, "up")))

        assert rep["ok"] is True
        assert rep["n_bougies"] == 400
        assert len(_FakeEnsemble.instances) == 1
        assert _FakeEnsemble.instances[0].trained is not None
        assert s.needs_retrain() is False   # état marqué

    def test_retrain_df_trop_petit_echec(self, tmp_path, monkeypatch):
        from deriv import ensemble_predictor, weekly_retrain

        _FakeEnsemble.instances = []
        monkeypatch.setattr(ensemble_predictor, "EnsemblePredictor",
                            _FakeEnsemble)
        s = self._sched(tmp_path)
        rep = asyncio.run(s.retrain_async(df=make_df(100, "up")))

        assert rep["ok"] is False
        assert "bougies" in rep["erreur"]
        assert s.needs_retrain() is True    # pas de mark sur échec
        assert _FakeEnsemble.instances == []

    def test_retrain_erreur_entrainement_capturee(self, tmp_path, monkeypatch):
        from deriv import ensemble_predictor, weekly_retrain

        class _BoomEnsemble:
            def train_all(self, df):
                raise RuntimeError("fit impossible")

        monkeypatch.setattr(ensemble_predictor, "EnsemblePredictor",
                            _BoomEnsemble)
        s = self._sched(tmp_path)
        rep = asyncio.run(s.retrain_async(df=make_df(400, "up")))
        assert rep["ok"] is False
        assert "entrainement" in rep["erreur"]

    def test_hook_bot_executor_present(self):
        """bot_executor instancie bien le scheduler (intégration mission)."""
        from deriv.bot_executor import DerivBotExecutor
        from deriv.weekly_retrain import RetrainScheduler

        bot = DerivBotExecutor(capital_usd=150.0)
        assert isinstance(bot.retrain, RetrainScheduler)
        assert isinstance(bot.retrain.needs_retrain(), bool)


# ── TESTS TIMEFRAMES SUPÉRIEURS (H1, H4) ────────────────────────────

class TestTimeframesSupérieurs:
    """
    Tests pour les timeframes longs (H1, H4) — API v4.
    Le bot doit adapter sa WFO selon la granularité.
    """

    @pytest.mark.parametrize("granularity,label", [
        (3600,  "H1"),
        (14400, "H4"),
        (900,   "M15"),
    ])
    def test_wfo_adaptatif_selon_granularite(self, granularity, label):
        """WalkForwardOptimizer s'adapte selon la granularité."""
        from deriv.walk_forward_optimizer import WalkForwardOptimizer
        wfo = WalkForwardOptimizer(granularity=granularity)

        if granularity >= 3600:
            assert wfo.TRAIN_SIZE <= 300
            assert wfo.EMBARGO <= 10
        else:
            assert wfo.TRAIN_SIZE >= 300

    def test_data_collector_granularite_h1(self):
        """Le data collector accepte granularity=3600."""
        from deriv.data_collector import DerivDataCollector
        c = DerivDataCollector("frxEURUSD", 3600)
        assert c.tick_size == 3600

    def test_requete_api_v4_structure(self):
        """La requête utilise subscribe=0 et adjust_start_time=1."""
        import inspect
        from deriv.data_collector import DerivDataCollector
        src = inspect.getsource(DerivDataCollector.get_candle_history)
        # Les paramètres sont passés en keyword args (subscribe=0, adjust_start_time=1)
        assert "subscribe" in src, "Paramètre 'subscribe' manquant dans get_candle_history"
        assert "adjust_start_time" in src, "Paramètre 'adjust_start_time' manquant"
        assert "ticks_history" in src or "candles" in src

    def test_count_pour_granularity_h1(self):
        """_count_pour_granularity retourne 1000 pour H1."""
        from deriv.symbol_optimizer import _count_pour_granularity
        assert _count_pour_granularity(3600) == 1000

    def test_count_pour_granularity_h4(self):
        """_count_pour_granularity retourne 500 pour H4."""
        from deriv.symbol_optimizer import _count_pour_granularity
        assert _count_pour_granularity(14400) == 500

    def test_count_pour_granularity_m1(self):
        """_count_pour_granularity retourne 3000 pour M1."""
        from deriv.symbol_optimizer import _count_pour_granularity
        assert _count_pour_granularity(60) == 3000

    def test_symbol_optimizer_contient_h1_h4(self):
        """SYMBOLES_A_TESTER contient au moins une combinaison H1 et H4."""
        from deriv.symbol_optimizer import SYMBOLES_A_TESTER
        granules = {gran for _, gran, _ in SYMBOLES_A_TESTER}
        assert 3600 in granules, "H1 (3600) manquant dans SYMBOLES_A_TESTER"
        assert 14400 in granules, "H4 (14400) manquant dans SYMBOLES_A_TESTER"

    def test_tester_symbole_count_adaptatif(self, monkeypatch):
        """tester_symbole utilise _count_pour_granularity quand count=None."""
        import deriv.symbol_optimizer as so
        from unittest.mock import AsyncMock, MagicMock, patch

        captured = {}

        async def fake_get_candle_history(count=None, target_count=None):
            c_val = count if count is not None else target_count
            captured["count"] = c_val
            np.random.seed(42)
            n = max(c_val or 1000, 700)
            return pd.DataFrame({
                "open": np.random.randn(n),
                "high": np.random.randn(n) + 1,
                "low": np.random.randn(n) - 1,
                "close": np.random.randn(n),
                "volume": np.random.randint(100, 1000, n).astype(float),
            })

        # Patch là où l'objet est utilisé (import local dans tester_symbole)
        with patch("deriv.data_collector.DerivDataCollector") as MockColl, \
             patch("deriv.walk_forward_optimizer.WalkForwardOptimizer") as MockWFO:
            MockColl.return_value.connect = AsyncMock()
            MockColl.return_value.close = AsyncMock()
            MockColl.return_value.get_candle_history = fake_get_candle_history
            MockColl.return_value.get_candle_history_full = fake_get_candle_history
            MockWFO.return_value.run.return_value = {
                "XGBoost": {
                    "win_rate_moyen": 0.55,
                    "win_rate_std": 0.05,
                    "n_folds": 5,
                    "n_trades_total": 200,
                }
            }

            import asyncio
            asyncio.run(so.tester_symbole("frxEURUSD", 3600, "EURUSD H1", count=None))

        assert captured.get("count") == 1000, \
            f"count attendu 1000 pour H1, reçu {captured.get('count')}"

    def test_pagination_produit_plus_de_bougies(self):
        """get_candle_history_full doit pouvoir retourner > 1000 bougies."""
        from deriv.data_collector import DerivDataCollector
        c = DerivDataCollector("frxEURUSD", 14400)
        # Vérifie que la méthode existe et accepte target_count
        import inspect
        assert hasattr(c, "get_candle_history_full")
        sig = inspect.signature(c.get_candle_history_full)
        assert "target_count" in sig.parameters


if __name__ == "__main__":
    raise SystemExit(
        f"python -m pytest {os.path.abspath(__file__)} -v --tb=short"
    )












