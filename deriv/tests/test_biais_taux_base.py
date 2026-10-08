"""

P0 — BIAIS DE TAUX DE BASE : tests de non-régression.

deriv/tests/test_biais_taux_base.py



Contexte (mesuré le 2026-09 sur CRASH500 M1, 45 folds) :

    WR annoncé      : XGBoost 89.67% / HMM 89.67% → « ✅ PRÊT DÉMO »

    taux de base UP : 89.67% (part de bougies haussières sur les fenêtres

                      de test des mêmes folds)

    => le « win rate » valait EXACTEMENT la dérive de l'indice : aucune

       compétence prédictive. Le modèle ne prédisait jamais « down ».

    Ensemble4sur5   : 0.0% sur 40 trades à CHAQUE fold (placeholder

                      `win_rate=0.0` de `_evaluate` jamais recalculé car

                      `pooled_raw` restait vide pour l'ensemble).



Ces tests verrouillent les 3 correctifs :

    1. un signal NEUTRE (vote 0.5 / proba 0.5) n'est plus converti en 0.5

       — sinon il est élu « UP » dès que le seuil tombe à 0.50 ;

    2. `_pooled_threshold` sélectionne le seuil sur la BALANCED ACCURACY

       (et refuse un seuil qui ne prédit qu'un seul sens) ;

    3. `demo_ready` exige un EDGE réel vs le taux de base ET des trades

       sur bougies BAISSIÈRES.



Aucun test n'instancie `WalkForwardOptimizer` (import lourd xgboost/

hmmlearn) : les méthodes testées sont statiques, ou appelées via

`__new__` — la suite reste rapide.



Lance avec : python -m pytest deriv/tests/test_biais_taux_base.py -v

"""

import ast

from pathlib import Path



import numpy as np

import pandas as pd

import pytest



from deriv.threshold_calibrator import calibrate_threshold

from deriv.walk_forward_optimizer import (

    MIN_DOWN_TRADES_DEMO,

    MIN_EDGE_DEMO,

    WalkForwardOptimizer,

    _EnsembleAdapter,

)




from deriv.client import DerivConnection
RACINE = Path(__file__).resolve().parents[2]

SOURCE_ENSEMBLE = RACINE / "deriv" / "ensemble_predictor.py"

SOURCE_WFO = RACINE / "deriv" / "walk_forward_optimizer.py"





def opt_sans_init(min_window: int = 60) -> WalkForwardOptimizer:

    """Instance sans __init__ (évite l'import lourd des 7 modèles)."""

    o = WalkForwardOptimizer.__new__(WalkForwardOptimizer)

    o.MIN_WINDOW = min_window

    return o





def make_prices(n: int = 100, up_ratio: float = 0.6, seed: int = 1) -> pd.DataFrame:

    """OHLCV minimal : seules les variations de `close` comptent."""

    rng = np.random.default_rng(seed)

    pas = np.where(rng.random(n) < up_ratio, 1.0, -1.0)

    pas = pas * rng.uniform(0.1, 1.0, n)

    close = 100.0 + np.cumsum(pas)

    return pd.DataFrame({

        "open": close, "high": close + 0.1, "low": close - 0.1,

        "close": close, "volume": 100.0,

    }, index=pd.date_range("2025-01-01", periods=n, freq="1min"))





# ══ 1. NORMALISATION DES SIGNAUX (statique) ═══════════════════════════



class TestNormalisationSignal:



    @pytest.mark.parametrize("valeur,attendu", [

        (0.0, 0.0), (1.0, 1.0), (0.7, 0.7), (0.42, 0.42),

    ])

    def test_to_proba_valides(self, valeur, attendu):

        assert WalkForwardOptimizer._to_proba(valeur) == attendu



    @pytest.mark.parametrize("valeur", [

        None, "0.7", True, False, np.nan, np.inf, -0.1, 1.5,

    ])

    def test_to_proba_inexploitables(self, valeur):

        assert WalkForwardOptimizer._to_proba(valeur) is None



    @pytest.mark.parametrize("vote,attendu", [

        (1, 1.0), (0, 0.0), (1.0, 1.0), (0.0, 0.0),

        (0.7, 0.7), (0.3, 0.3), (0.8, 0.8), (0.2, 0.2),

    ])

    def test_vote_to_proba_directionnels(self, vote, attendu):

        assert WalkForwardOptimizer._vote_to_proba(vote) == attendu



    @pytest.mark.parametrize("vote", [

        None, "1", True, False, np.nan, 0.5, 0.4, 0.6, 0.31, 0.69,

    ])

    def test_vote_neutre_ou_invalide_rejete(self, vote):

        """Un vote neutre ne porte AUCUNE information → rejeté (None).



        C'est LA correction du bug CRASH500 : convertir 0.5 en proba 0.5

        faisait élire « UP » en permanence (`proba >= thr`) et le win rate

        mesurait la dérive de l'indice au lieu de la compétence.

        """

        assert WalkForwardOptimizer._vote_to_proba(vote) is None





# ══ 2. WIN RATE / BALANCED ACCURACY ═══════════════════════════════════



class TestApplyThreshold:



    def test_toujours_up_wr_eleve_mais_balanced_50(self):

        """Reproduction CRASH500 : 89.67% de bougies haussières.



        Un modèle qui prédit TOUJOURS « UP » obtient un win rate de

        89.67% — indiscernable d'une performance tant qu'on ne mesure pas

        le rappel séparément sur les deux classes.

        """

        rng = np.random.default_rng(0)

        n = 4455

        actuals = (rng.random(n) < 0.8967).astype(int)

        probas = np.full(n, 0.9)                     # toujours UP

        res = WalkForwardOptimizer._apply_threshold(probas, actuals, 0.7, 0.3)



        assert res["n_trades"] == n

        assert res["win_rate"] == pytest.approx(actuals.mean(), abs=1e-9)

        assert res["win_rate"] > 0.85                 # « performance » ?!

        assert res["balanced_acc"] == pytest.approx(0.5, abs=1e-9)

        assert res["n_down_reel"] > 0                 # le biais est visible



    def test_modele_competent_balanced_eleve(self):

        actuals = np.array([1, 1, 1, 0, 0, 0, 1, 0])

        probas = np.array([0.9, 0.8, 0.75, 0.1, 0.2, 0.25, 0.85, 0.15])

        res = WalkForwardOptimizer._apply_threshold(probas, actuals, 0.7, 0.3)

        assert res["win_rate"] == 1.0

        assert res["balanced_acc"] == 1.0



    def test_masque_vide(self):

        res = WalkForwardOptimizer._apply_threshold(

            np.full(10, 0.55), np.zeros(10, dtype=int), 0.7, 0.3

        )

        assert res["n_trades"] == 0

        assert res["win_rate"] == 0.0

        assert res["balanced_acc"] == 0.5



    def test_une_seule_classe_presente_balanced_neutre(self):

        """Indice 100% haussier : la balanced accuracy n'est pas identifiable."""

        res = WalkForwardOptimizer._apply_threshold(

            np.full(20, 0.9), np.ones(20, dtype=int), 0.7, 0.3

        )

        assert res["balanced_acc"] == 0.5

        assert res["n_down_reel"] == 0





# ══ 3. CALIBRATION POOLÉE ANTI-BIAIS ══════════════════════════════════



class TestPooledThreshold:



    def test_refuse_de_calibrer_un_sens_unique(self):

        """Pool « toujours UP » → aucun candidat exploitable → repli neutre.



        Ancien comportement : le seuil 0.50 maximisait le WR brut (89.67%)

        et était retenu. Correctif : un candidat qui ne prédit qu'un seul

        sens (< 5 trades de l'autre côté) est IGNORÉ.

        """

        rng = np.random.default_rng(3)

        actuals = (rng.random(300) < 0.9).astype(int)

        probas = np.full(300, 0.95)

        thr_up, thr_down = WalkForwardOptimizer._pooled_threshold(probas, actuals)

        assert (thr_up, thr_down) == (0.60, 0.40)     # repli, jamais 0.50

        assert thr_up == pytest.approx(1 - thr_down)



    def test_pool_separable_retrouve_le_seuil(self):

        probas = np.array([0.9] * 100 + [0.1] * 100)

        actuals = np.array([1] * 100 + [0] * 100)

        thr_up, thr_down = WalkForwardOptimizer._pooled_threshold(probas, actuals)

        res = WalkForwardOptimizer._apply_threshold(probas, actuals, thr_up, thr_down)

        assert res["balanced_acc"] == 1.0

        assert res["n_up_reel"] == 100 and res["n_down_reel"] == 100



    def test_masque_insuffisant_ignore(self):

        """Moins de 10 prédictions retenues → candidat ignoré."""

        probas = np.array([0.9, 0.1, 0.9, 0.1, 0.9])

        actuals = np.array([1, 0, 1, 0, 1])

        thr_up, thr_down = WalkForwardOptimizer._pooled_threshold(probas, actuals)

        assert (thr_up, thr_down) == (0.60, 0.40)





# ══ 4. VERDICT BIAIS ══════════════════════════════════════════════════



class TestAssessBiais:



    def test_crash500_toujours_up_est_biaise(self):

        """WR 89.67% = taux de base 89.67% → aucune compétence."""

        b = WalkForwardOptimizer._assess_bias(

            win_rate=0.8967, taux_base=0.8967, balanced_wr=0.5,

            n_up_reel=3995, n_down_reel=460,

        )

        assert b["hors_biais_taux_base"] is False

        assert b["edge_vs_taux_base"] == 0.0

        assert "balanced accuracy" in b["raison"]



    def test_competence_reelle_validee(self):

        b = WalkForwardOptimizer._assess_bias(

            win_rate=0.62, taux_base=0.50, balanced_wr=0.61,

            n_up_reel=200, n_down_reel=200,

        )

        assert b["hors_biais_taux_base"] is True

        assert b["edge_vs_taux_base"] == pytest.approx(0.12, abs=1e-9)



    def test_aucun_trade_baissier_rejete(self):

        """Performance validée dans un seul régime de marché → rejetée."""

        b = WalkForwardOptimizer._assess_bias(

            win_rate=0.90, taux_base=0.60, balanced_wr=0.70,

            n_up_reel=500, n_down_reel=MIN_DOWN_TRADES_DEMO - 1,

        )

        assert b["hors_biais_taux_base"] is False

        assert "baissières" in b["raison"]



    def test_edge_insuffisant_rejete(self):

        b = WalkForwardOptimizer._assess_bias(

            win_rate=0.55, taux_base=0.52, balanced_wr=0.55,

            n_up_reel=300, n_down_reel=100,

        )

        assert b["hors_biais_taux_base"] is False

        assert b["edge_vs_taux_base"] < MIN_EDGE_DEMO

        assert "taux de base" in b["raison"]



    def test_edge_limite_inclusif(self):

        b = WalkForwardOptimizer._assess_bias(

            win_rate=0.50 + MIN_EDGE_DEMO, taux_base=0.50, balanced_wr=0.51,

            n_up_reel=100, n_down_reel=100,

        )

        assert b["hors_biais_taux_base"] is True





# ══ 5. _evaluate : LE NEUTRE N'EST PAS UN TRADE ═══════════════════════



class TestEvaluateIgnoreNeutre:



    def _attendu_up(self, df: pd.DataFrame, min_window: int) -> list[int]:

        return [

            int(df["close"].iloc[i] > df["close"].iloc[i - 1])

            for i in range(min_window, len(df))

        ]



    def test_modele_neutre_ne_trade_pas(self):

        """Un modèle qui ne dit que « neutre » ne produit AUCUN trade.



        Avant correctif : vote 0.5 → proba 0.5 → élu « UP » → win rate

        égal au taux de base des bougies haussières (faux positif).

        """

        o = opt_sans_init()

        df = make_prices(100, up_ratio=0.9, seed=4)

        res = o._evaluate(lambda w: {"vote": 0.5}, df)



        assert res is not None

        assert res["n_trades"] == 0

        assert res["probas"] == []

        assert res["aucun_signal"] is True

        # Le taux de base reste publié pour le diagnostic

        assert res["up_rate"] == pytest.approx(

            float(np.mean(self._attendu_up(df, o.MIN_WINDOW))), abs=1e-9

        )



    def test_proba_up_exploitee(self):

        o = opt_sans_init()

        df = make_prices(100, up_ratio=0.7, seed=5)

        res = o._evaluate(lambda w: {"proba_up": 0.9}, df)



        assert res["n_trades"] == 100 - o.MIN_WINDOW

        assert all(p == 0.9 for p in res["probas"])

        attendu = self._attendu_up(df, o.MIN_WINDOW)

        assert res["actuals"] == attendu

        assert res["up_rate"] == pytest.approx(float(np.mean(attendu)), abs=1e-9)



    def test_p_useful_prioritaire_sur_proba_up(self):

        """HMM expose p_useful : elle prime sur proba_up (échelle différente)."""

        o = opt_sans_init()

        df = make_prices(80, up_ratio=0.5, seed=6)

        res = o._evaluate(

            lambda w: {"p_useful": 0.2, "proba_up": 0.9, "vote": 1.0}, df

        )

        assert res["n_trades"] == 80 - o.MIN_WINDOW

        assert all(p == 0.2 for p in res["probas"])



    def test_vote_directionnel_utilise_en_dernier_recours(self):

        o = opt_sans_init()

        df = make_prices(80, up_ratio=0.5, seed=7)

        res = o._evaluate(lambda w: {"vote": 1.0}, df)

        assert res["n_trades"] == 80 - o.MIN_WINDOW

        assert all(p == 1.0 for p in res["probas"])



    def test_prediction_en_erreur_ignoree(self):

        o = opt_sans_init()

        df = make_prices(75, up_ratio=0.5, seed=8)



        def predict_instable(window):

            if len(window) % 2 == 0:

                raise RuntimeError("panne simulée")

            return {"proba_up": 0.8}



        res = o._evaluate(predict_instable, df)

        assert 0 < res["n_trades"] < 75 - o.MIN_WINDOW

        assert all(p == 0.8 for p in res["probas"])

# ══ 6. ÉCHELLE UNIFIÉE : ENSEMBLE + HMM ═══════════════════════════════



class _EnsembleFactice:

    def __init__(self, signal):

        self._signal = signal



    def predict(self, df):

        return {"signal": self._signal}





class TestEnsembleAdapter:



    def test_buy_et_sell_alignes_sur_wrapper(self):

        """Même échelle que _StatelessWrapper (0.8 / 0.2) → seuil comparable."""

        buy = _EnsembleAdapter(_EnsembleFactice("BUY")).predict(None)

        sell = _EnsembleAdapter(_EnsembleFactice("SELL")).predict(None)

        assert (buy["vote"], buy["proba_up"]) == (1.0, 0.8)

        assert (sell["vote"], sell["proba_up"]) == (0.0, 0.2)



    def test_hold_n_expose_aucune_proba(self):

        """HOLD n'est pas un trade : 0.5 serait élu « UP » par le seuil 0.50."""

        hold = _EnsembleAdapter(_EnsembleFactice("HOLD")).predict(None)

        assert hold["proba_up"] is None

        assert hold["p_useful"] is None

        assert WalkForwardOptimizer._to_proba(hold["proba_up"]) is None



    @pytest.mark.parametrize("signal", ["hold", "Hold", "", "INCONNU"])

    def test_signal_non_directionnel_neutre(self, signal):

        res = _EnsembleAdapter(_EnsembleFactice(signal)).predict(None)

        assert res["proba_up"] is None





class TestHMMExposeProbaUp:



    @staticmethod

    def _predict_returns() -> list[ast.Dict]:

        arbre = ast.parse(SOURCE_ENSEMBLE.read_text(encoding="utf-8"))

        for noeud in ast.walk(arbre):

            if isinstance(noeud, ast.ClassDef) and noeud.name == "HMMPredictor":

                for item in noeud.body:

                    if (isinstance(item, ast.FunctionDef)

                            and item.name == "predict"):

                        return [n for n in ast.walk(item) if isinstance(n, ast.Dict)]

        raise AssertionError("HMMPredictor.predict introuvable")



    def test_predict_retourne_proba_up(self):

        """Sans cette clé, `_evaluate` retombait sur 0.5 → WR = taux de base

        (bug mesuré : HMM 89.67% = XGBoost 89.67%, aucune compétence)."""

        cles = {k.value for d in self._predict_returns()

                for k in d.keys if isinstance(k, ast.Constant)}

        assert "proba_up" in cles

        assert "vote" in cles



    def test_predict_neutre_sans_modele(self):

        from deriv.ensemble_predictor import HMMPredictor



        hmm = HMMPredictor.__new__(HMMPredictor)   # pas de _load() disque

        hmm.model, hmm.is_trained, hmm.scaler, hmm.labels = None, False, None, {}

        res = hmm.predict(make_prices(80))

        assert res["vote"] == 0.5

        assert res["p_useful"] is None





# ══ 7. ALIGNEMENT probas/actuals — INDEXERROR DE FIN DE WFO ═══════════



class TestAlignementPool:



    def test_aucun_signal_probas_actuals_alignes(self):

        """Fold sans signal : `probas` ET `actuals` vides (mêmes longueurs).



        Avant correctif, `actuals` recevait les 99 bougies du fold alors que

        `probas` était vide : le pool cumulé de `run()` finissait désaligné

        et `_pooled_threshold` plantait par IndexError APRÈS les 45 folds

        (mesuré : 642 probas vs 842 actuals = 2 folds sans signal).

        """

        o = opt_sans_init()

        res = o._evaluate(lambda w: {"vote": 0.5}, make_prices(100, seed=4))



        assert res["aucun_signal"] is True

        assert res["actuals"] == []

        assert len(res["probas"]) == len(res["actuals"]) == 0

        # Le taux de base du fold reste disponible pour le diagnostic

        assert 0.0 <= res["up_rate"] <= 1.0



    def test_probas_actuals_toujours_alignes(self):

        """Invariant de `_evaluate` : len(probas) == len(actuals) == n_trades."""

        o = opt_sans_init()

        df = make_prices(120, up_ratio=0.6, seed=9)

        for predict in (

            lambda w: {"vote": 0.5},        # aucun signal exploitable

            lambda w: {"proba_up": 0.9},    # signal franc

            lambda w: {"vote": 1.0},        # vote directionnel de secours

        ):

            res = o._evaluate(predict, df)

            assert len(res["probas"]) == len(res["actuals"])

            assert res["n_trades"] == len(res["probas"])



    def test_pool_cumule_reste_alignable(self):

        """Simule la collecte poolée de `run()` sur des folds mixtes."""

        o = opt_sans_init()

        pooled = {"probas": [], "actuals": []}

        for seed in (11, 12, 13):

            df = make_prices(100, up_ratio=0.5, seed=seed)

            predict = ((lambda w: {"vote": 0.5}) if seed == 12

                       else (lambda w: {"proba_up": 0.8}))

            res = o._evaluate(predict, df)

            if res["probas"]:          # garde-fou identique à celui de run()

                pooled["probas"].extend(res["probas"])

                pooled["actuals"].extend(res["actuals"])



        assert len(pooled["probas"]) == len(pooled["actuals"])

        assert len(pooled["probas"]) == 2 * (100 - o.MIN_WINDOW)

        thr_up, thr_down = o._pooled_threshold(pooled["probas"], pooled["actuals"])

        assert 0.50 <= thr_up <= 0.75

        assert thr_down == pytest.approx(1 - thr_up, abs=1e-9)



    def test_pooled_threshold_tronque_au_lieu_de_planter(self):

        """Garde-fou : troncature sur la longueur commune, jamais d'exception."""

        o = opt_sans_init()

        probas = [0.9] * 30 + [0.1] * 30            # 60 probas

        y_true = [1] * 30 + [0] * 30 + [1] * 20     # 80 actuals (désaligné)



        thr_up, thr_down = o._pooled_threshold(probas, y_true)



        assert 0.50 <= thr_up <= 0.75

        assert thr_down == pytest.approx(1 - thr_up, abs=1e-9)



    def test_collecte_pool_stateless_gardee(self):

        """`run()` ne doit étendre `actuals` que si `probas` l'est aussi."""

        src = SOURCE_WFO.read_text(encoding="utf-8")

        # Ancre tolérante : la boucle stateless de run() itère soit

        # directement sur self.stateless_models, soit sur le dict local

        # `stateless` (mode crash : CrashGuard y est ajouté).

        for ancre in (

            "for name, ModelClass in stateless.items()",

            "for name, ModelClass in self.stateless_models.items()",

        ):

            try:

                depart = src.index(ancre)

                break

            except ValueError:

                continue

        else:

            pytest.fail("boucle des modèles stateless introuvable dans run()")

        bloc = src[depart:src.index("# ENSEMBLE complet", depart)]

        assert 'if res["probas"]:' in bloc

class TestCalibrateThresholdDirect:
    """Tests directes de deriv.threshold_calibrator.calibrate_threshold."""

    def test_empty_arrays_return_fallback(self):
        thr_up, thr_down = calibrate_threshold([], [], min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_none_inputs_return_fallback(self):
        thr_up, thr_down = calibrate_threshold(None, [1, 0], min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_all_same_class_returns_fallback(self):
        probas = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        actuals = np.array([1] * 8)
        thr_up, thr_down = calibrate_threshold(probas, actuals, min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_identical_probas_return_fallback(self):
        probas = np.array([0.5] * 100)
        actuals = np.array([0, 1] * 50)
        thr_up, thr_down = calibrate_threshold(probas, actuals, min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_min_trades_zero_uses_all_candidates(self):
        probas = np.array([0.9, 0.1, 0.9, 0.1])
        actuals = np.array([1, 0, 1, 0])
        thr_up, thr_down = calibrate_threshold(probas, actuals, min_trades=0)
        assert thr_up == pytest.approx(0.55)
        assert thr_down == pytest.approx(0.45)

    def test_single_candidate(self):
        probas = np.array([0.9, 0.1, 0.9, 0.1])
        actuals = np.array([1, 0, 1, 0])
        thr_up, thr_down = calibrate_threshold(probas, actuals, candidates=[0.60], min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_best_score_zero_boundary(self):
        probas = np.array([0.9, 0.1, 0.9, 0.1])
        actuals = np.array([0, 1, 0, 1])
        thr_up, thr_down = calibrate_threshold(probas, actuals, min_trades=1)
        assert (thr_up, thr_down) == (0.60, 0.40)

    def test_threshold_clamped_to_075(self):
        probas = np.array([0.9] * 20 + [0.1] * 20)
        actuals = np.array([1] * 20 + [0] * 20)
        thr_up, thr_down = calibrate_threshold(probas, actuals, candidates=[0.80, 0.90], min_trades=1)
        assert thr_up == 0.75
        assert thr_down == 0.25

    def test_extract_token_from_url(self):
        url = "https://bot.deriv.com/callback#access_token=TOKEN123&expires_in=86400"
        assert DerivConnection.extract_token_from_url(url) == "TOKEN123"
        assert DerivConnection.extract_token_from_url("https://bot.deriv.com/callback") is None

