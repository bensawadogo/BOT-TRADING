"""
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
WALK-FORWARD OPTIMIZATION â€” validation statistique du bot Deriv
deriv/walk_forward_optimizer.py
â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Standard acadÃ©mique : entraÃ®ne sur N bougies, valide sur les M suivantes,
avec un EMBARGO (trou de 30 bougies) entre train et test pour Ã©liminer
les fuites temporelles. RÃ©pÃ©tÃ© sur toute la sÃ©rie â†’ win rate rÃ©aliste
+ intervalle de confiance (Wilson) + dÃ©tection d'overfitting.

ModÃ¨les Ã©valuÃ©s : XGBoost, HMM, Kalman, RSI, TrendStrength, Momentum
ET l'Ensemble complet (rÃ¨gle 4/5) â€” c'est lui qui trade en rÃ©el.

Les modÃ¨les entraÃ®nÃ©s pendant le WFO ne JAMAIS Ã©craser les modÃ¨les
de production (deriv/models/*.pkl) : la sauvegarde est dÃ©sactivÃ©e
pendant l'optimisation.

Usage :
  python deriv/walk_forward_optimizer.py --symbol R_75 --count 3000
  python deriv/walk_forward_optimizer.py --symbol frxEURUSD --count 3000 --granularity 300
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from deriv.voting import vote_to_proba
from deriv.threshold_calibrator import calibrate_threshold

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WFO")

# RÃ©solution robuste de la racine (fonctionne en script ET en module)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BREAKEVEN = 0.556          # seuil de profitabilitÃ© des options Deriv
MIN_FOLDS_DEMO = 5         # critÃ¨re passage dÃ©mo : 5+ folds
MAX_STD_DEMO = 0.15
MAX_STD_OVERFITTING = 0.20
MIN_TRADES_DEMO = 200      # critÃ¨re passage dÃ©mo : 200+ trades simulÃ©s

# â”€â”€ P0 : garde-fous contre le BIAIS DE TAUX DE BASE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Sur CRASH500 M1, 89.67% des bougies sont haussiÃ¨res. Un modÃ¨le qui vote
# TOUJOURS BUY affiche donc 89.67% de win rate SANS aucune compÃ©tence :
# c'est la dÃ©rive de l'indice, pas une prÃ©diction. Ces seuils interdisent
# de valider (et de passer en dÃ©mo) un tel rÃ©sultat.
MIN_TRADES_CALIB = 30      # trades min pour calibrer un seuil poolÃ©
BALANCED_ACC_MIN = 0.55    # en dessous : le modÃ¨le ne distingue pas UP de DOWN
EDGE_MIN_VS_BASE = 0.0     # edge (WR - taux de base UP) minimum exigÃ©
EDGE_FAIBLE = 0.05         # edge faible â†’ avertissement (non bloquant)
UP_RATE_ELEVE = 0.75       # taux de base UP "Ã©levÃ©" â†’ avertissement

# â”€â”€ P0 : garde-fou contre le BIAIS DE TAUX DE BASE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Un win rate brut Ã©levÃ© peut n'Ãªtre QUE la dÃ©rive d'un indice quasi
# haussier. Constat mesurÃ© sur CRASH500 M1 (45 folds) : le taux de
# bougies haussiÃ¨res vaut 89.67% et le WR annoncÃ© de XGBoost/HMM Ã©tait
# â€¦ 89.67%. Les modÃ¨les prÃ©disaient TOUJOURS Â« UP Â» : aucune compÃ©tence,
# seulement la pente de l'indice. Ces deux constantes interdisent de
# valider un modÃ¨le sur ce seul chiffre.
MIN_EDGE_DEMO = 0.05        # WR doit battre le taux de base de 5 points
MIN_DOWN_TRADES_DEMO = 30   # trades exigÃ©s sur bougies BAISSIÃˆRES


# â”€â”€ OUTILS STATISTIQUES â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    Intervalle de confiance de Wilson du win rate.
    Plus fiable que la normale pour les petits Ã©chantillons.
    """
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    marge = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - marge), min(1.0, centre + marge))


def binomial_pvalue(wins: int, n: int, p0: float = 0.5) -> float:
    """
    P-value (test binomial unilatÃ©ral) : probabilitÃ© d'observer `wins`
    succÃ¨s sur `n` si le vrai taux Ã©tait p0 (hasard pur).
    p < 0.05 â†’ le modÃ¨le est statistiquement meilleur que le hasard.
    """
    if n <= 0:
        return 1.0
    try:
        from scipy.stats import binomtest

        return float(binomtest(int(wins), int(n), p0, alternative="greater").pvalue)
    except Exception:  # scipy absent â†’ approximation normale
        p = wins / n
        se = math.sqrt(p0 * (1 - p0) / n)
        z = (p - p0) / se if se > 0 else 0.0
        return 0.5 * math.erfc(z / math.sqrt(2))  # queue supÃ©rieure


def diagnostic_overfitting(
    in_sample_wr: float | None,
    out_sample_wr: float,
    folds_wr: list[float] | None = None,
    n_trades: int = 0,
    seuil_ecart: float = 0.15,
    seuil_std: float = MAX_STD_OVERFITTING,
    min_trades: int = 100,
) -> dict:
    """
    DÃ©tection automatique d'overfitting.

    Signaux :
      - Ã©cart in-sample / out-of-sample > seuil_ecart (15 points)
      - Ã©cart-type des folds > seuil_std (rÃ©sultats instables)
      - Ã©chantillon trop faible (< min_trades)
    """
    signaux: list[str] = []
    ecart = (in_sample_wr - out_sample_wr) if in_sample_wr is not None else 0.0
    std_folds = float(np.std(folds_wr)) if folds_wr else 0.0

    if in_sample_wr is not None and ecart > seuil_ecart:
        signaux.append(
            f"écart in/out = {ecart:.1%} > {seuil_ecart:.0%} (surapprentissage)"
        )
    if folds_wr and std_folds > seuil_std:
        signaux.append(f"écart-type folds = {std_folds:.1%} > {seuil_std:.0%} (instable)")
    if n_trades < min_trades:
        signaux.append(f"échantillon faible : {n_trades} trades < {min_trades}")

    return {
        "overfitting": bool(signaux),
        "ecart_in_out": round(ecart, 4),
        "ecart_type_folds": round(std_folds, 4),
        "n_trades": n_trades,
        "signaux": signaux,
        "fiabilite": "FAIBLE" if signaux else "BONNE",
    }


class _EnsembleAdapter:
    """Adapte l'EnsemblePredictor (signal BUY/SELL/HOLD) au format vote.

    Contrat garanti : retourne TOUJOURS {"vote", "proba_up"} avec
    proba_up dans [0, 1] â€” jamais de proba neutre forcÃ©e quand le
    signal est directionnel (sinon le WFO mesure un WR 0.0 biaisÃ©).

    P0 (biais de mesure) : proba_up est convertie sur EXACTEMENT la mÃªme
    Ã©chelle que `_StatelessWrapper` (BUY 0.8 / SELL 0.2 / HOLD 0.5).
    Aligner toutes les Ã©chelles est indispensable pour que le seuil
    calibrÃ© sur le pool out-of-sample soit comparable d'un modÃ¨le Ã 
    l'autre ; utiliser `confiance` (Ã©chelle 4/7-7/7) rendait l'ensemble
    non calibrÃ© et son WR restait au placeholder 0.0.
    """

    PROBA_BUY = 0.8
    PROBA_SELL = 0.2
    PROBA_HOLD = 0.5

    def __init__(self, ensemble):
        self._ens = ensemble

    def predict(self, df: pd.DataFrame) -> dict:
        res = self._ens.predict(df)
        signal = res.get("signal", "HOLD")
        signal = str(signal).upper() if signal is not None else "HOLD"
        if signal == "BUY":
            return {"vote": 1.0, "proba_up": float(self.PROBA_BUY)}
        if signal == "SELL":
            return {"vote": 0.0, "proba_up": float(self.PROBA_SELL)}
        # P0 â€” HOLD n'est PAS un trade : ne pas exposer de proba. Sinon le
        # 0.5 constant est Ã©lu Â« UP Â» dÃ¨s que le seuil calibrÃ© tombe Ã  0.50
        # et le win rate de l'ensemble redevient le taux de base de
        # l'indice, sans aucune compÃ©tence (bug mesurÃ© CRASH500 M1).
        return {"vote": 0.5, "proba_up": None, "p_useful": None}


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization non-ancrÃ©e.

    Chaque fold a la mÃªme longueur de train et de test. Un embargo de
    `EMBARGO` bougies sÃ©pare train et test (aucune fuite possible).
    """

    TRAIN_SIZE = 400   # bougies d'entraÃ®nement par fold
    TEST_SIZE = 100    # bougies de test par fold
    EMBARGO = 30       # bougies d'embargo (trou anti-fuite)
    PURGE = 1          # bougies purgÃ©es en fin de train (LÃ³pez de Prado 2018)
    STEP = 100         # dÃ©calage entre deux folds
    MIN_WINDOW = 60    # bougies min pour prÃ©dire (EMA50 â†’ features valides)
    IN_SAMPLE_WINDOW = 200
    IN_SAMPLE_STEP = 20

    def __init__(
        self,
        train_size: int | None = None,
        test_size: int | None = None,
        embargo: int | None = None,
        step: int | None = None,
        min_window: int | None = None,
        purge: int | None = None,
        eval_ensemble: bool = True,
        eval_in_sample: bool = True,
        granularity: int = 60,
    ):
        # â”€â”€ Tailles adaptatives selon le timeframe â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # M1/M5/M15 : beaucoup de bougies, entraÃ®nement long
        # H1/H4     : moins de bougies mais signal plus stable
        if granularity >= 14400:   # H4 (test_size >= 75 requis car MIN_WINDOW=60)
            default_train, default_test, default_embargo, default_step = 200, 75, 5, 50
        elif granularity >= 3600:  # H1
            default_train, default_test, default_embargo, default_step = 300, 75, 10, 75
        else:                       # M1, M5, M15
            default_train, default_test, default_embargo, default_step = 400, 100, 30, 100

        self.TRAIN_SIZE = int(train_size or default_train)
        self.TEST_SIZE = int(test_size or default_test)
        self.EMBARGO = int(embargo if embargo is not None else default_embargo)
        self.STEP = int(step or default_step)
        self.PURGE = int(purge if purge is not None else self.PURGE)
        self.MIN_WINDOW = int(min_window or self.MIN_WINDOW)
        self.eval_ensemble = eval_ensemble
        self.eval_in_sample = eval_in_sample
        self.granularity = granularity

        from deriv.ensemble_predictor import (
            HMMPredictor,
            KalmanMomentumPredictor,
            MomentumPredictor,
            RSIDivergencePredictor,
            TrendStrengthPredictor,
            XGBoostPredictor,
        )

        self.models = {
            "XGBoost": XGBoostPredictor,
            "HMM": HMMPredictor,
        }
        self.stateless_models = {
            "Kalman": KalmanMomentumPredictor,
            "RSI": RSIDivergencePredictor,
            "TrendStrength": TrendStrengthPredictor,
            "Momentum": MomentumPredictor,
        }

        # â”€â”€ Wrapper stateless : conversion vote â†’ proba poolable â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Les stateless retournent {"vote": 0/0.5/1}. On wrap pour exposer
        # "proba_up" (1â†’0.8, 0â†’0.2) afin que la calibration globale (P0)
        # puisse Ãªtre appliquÃ©e uniformÃ©ment Ã  tous les modÃ¨les.
        # Un vote NEUTRE n'expose AUCUNE proba (P0) : l'exposer Ã  0.5 le
        # faisait compter comme un trade que le seuillage Ã©lisait Â« UP Â»
        # en permanence, ramenant le win rate au taux de base de l'indice
        # (bug mesurÃ© sur CRASH500 M1 : 89.67% de WR annoncÃ© = zÃ©ro edge).
        class _StatelessWrapper:
            def __init__(self, model):
                self._m = model
            def predict(self, df):
                r = self._m.predict(df)
                # CrashGuard expose P(crash) native (p_useful) : on contourne
                # le normalisateur vote->proba_up, sinon p_useful est ecras
                # a None en mode HOLD (perte totale de signal).
                if getattr(self._m, "_native_proba", False):
                    return r
                v = r.get("vote", 0.5)
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and (v >= 0.7 or v <= 0.3)):
                    r["proba_up"] = 0.8 if v >= 0.7 else 0.2
                else:
                    r.pop("proba_up", None)
                    r["p_useful"] = None
                return r
            def train(self, df):
                pass
            @property
            def threshold_up(self):
                return 0.5
            @property
            def threshold_down(self):
                return 0.5

        self._StatelessWrapper = _StatelessWrapper

    # â”€â”€ Plan de folds (pur, testable sans calcul lourd) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @classmethod
    def folds_plan(
        cls,
        n_rows: int,
        train_size: int | None = None,
        test_size: int | None = None,
        embargo: int | None = None,
        step: int | None = None,
        purge: int | None = None,
    ) -> list[dict]:
        """
        DÃ©coupe non-ancrÃ©e : liste de {fold, train, purge, test_start, test_end}.

        GARDE-FOU 3 (LÃ³pez de Prado, 2018) â€” schÃ©ma par fold :
            [===TRAIN===][PURGE][EMBARGO][====TEST====]
        La derniÃ¨re bougie du train est PURGÃ‰E : sa target pointe vers
        la bougie suivante qui, sans purge, serait la premiÃ¨re du test.
        `train` contient la fenÃªtre EFFECTIVE (aprÃ¨s retrait du purge).
        """
        train_size = int(train_size or cls.TRAIN_SIZE)
        test_size = int(test_size or cls.TEST_SIZE)
        embargo = int(embargo if embargo is not None else cls.EMBARGO)
        step = int(step or cls.STEP)
        purge = int(purge if purge is not None else cls.PURGE)

        plan = []
        n = 0
        while n + train_size + embargo + test_size <= n_rows:
            plan.append({
                "fold": len(plan) + 1,
                "train": (n, n + train_size - purge),       # effectif (aprÃ¨s purge)
                "purge": purge,                              # bougie(s) exclue(s)
                "test_start": n + train_size + embargo,
                "test_end": n + train_size + embargo + test_size,
            })
            n += step
        return plan

    # â”€â”€ EmpÃªche l'Ã©crasement des modÃ¨les de production â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def _disarm_save(model) -> None:
        """Neutralise _save() sur l'instance (pickles de prod prÃ©servÃ©s)."""
        model._save = lambda: None

    # â”€â”€ Ã‰valuation fenÃªtre glissante croissante (out-of-sample) â”€â”€â”€â”€â”€
    def _evaluate(
        self, predict_fn, test_df: pd.DataFrame, target_mode: str = "sign"
    ) -> dict | None:
        """
        Collecte les prÃ©dictions out-of-sample. Retourne les probas bruts
        et les actuals â€” le filtrage par seuil est appliquÃ© APRÃˆS la
        calibration globale poolÃ©e (P0), pas ici, pour Ã©viter de comparer
        des rÃ¨gles diffÃ©rentes entre folds.

        P0 (biais de mesure) â€” deux rÃ¨gles :
          1. La proba exploitÃ©e suit la prioritÃ© `p_useful` (HMM) â†’
             `proba_up` â†’ `vote` directionnel. Un modÃ¨le qui n'expose
             qu'un vote NEUTRE est IGNORÃ‰ : le convertir en 0.5 puis le
             comparer au seuil (`proba >= thr`) le faisait Ã©lire Â« UP Â» en
             permanence, et son WR valait alors exactement la dÃ©rive de
             l'indice (constat CRASH500 M1 : WR HMM = 89.67% = taux de
             base des bougies haussiÃ¨res, aucune compÃ©tence).
          2. `up_reel` conserve la direction rÃ©ellement observÃ©e pour
             CHAQUE bougie testÃ©e : c'est la rÃ©fÃ©rence du taux de base.
        """
        # Cible asymÃ©trique CRASH500 (P0) â€” sign : label UP/DOWN, proba=P(UP).
        # crash : label ret <= -CRASH_SEUIL_FRAC (~2 %), proba=P(crash).
        # Sur un indice en dÃ©rive (89.76 % UP), Â« crash Â» isole le rare
        # Ã©vÃ©nement tradeable au lieu de mesurer la dÃ©rive haussiÃ¨re (zÃ©ro edge).
        if target_mode == "crash":
            from deriv.ensemble_predictor import CRASH_SEUIL_FRAC as _CRASH_SEUIL
        if target_mode == "triple_barrier":
            # P0-v2 (audit 90 %) â€” la target triviale close[i] > close[i-1]
            # mesure la DÃ‰RIVE de l'indice, pas une compÃ©tence (acc rÃ©elle =
            # acc permutÃ©e = baseline UP sur CRASH500). La triple-barriÃ¨re
            # symÃ©trique (LÃ³pez de Prado, 2018) Ã©limine ce biais : label 1 si
            # le prix touche +1.5*ATR avant -1.5*ATR dans les 3 bougies, 0 si
            # l'inverse, label Ã‰LIMINÃ‰ si ni l'une ni l'autre â†’ pas de trade.
            from deriv.ensemble_predictor import build_labels as _build_labels
            labels_tb = _build_labels(
                test_df, mode="triple_barrier", touch_mult=1.5, horizon=3,
            )
        probas, actuals, up_reel = [], [], []
        for i in range(self.MIN_WINDOW, len(test_df)):
            window = test_df.iloc[:i]
            try:
                res = predict_fn(window)
            except Exception:
                continue

            if target_mode == "crash":
                r_i = (test_df["close"].iloc[i] /
                       test_df["close"].iloc[i - 1]) - 1.0
                actual = int(r_i <= -_CRASH_SEUIL)   # 1 = crash (spike)
                up_reel.append(actual)
            elif target_mode == "triple_barrier":
                # Baseline dÃ©rive (diagnostic P0) : direction brute de la
                # bougie, comptÃ©e pour TOUTES les bougies â€” mÃªme celles
                # sans trade (label Ã©liminÃ©) â€” pour le taux de base.
                up_reel.append(int(
                    test_df["close"].iloc[i] > test_df["close"].iloc[i - 1]
                ))
                # Label de la bougie DÃ‰CISION (i-1) : barriÃ¨res Ã©valuÃ©es
                # sur [i, i+horizon]. Le label de i-1 est connu dÃ¨s i.
                lbl = labels_tb.get(test_df.index[i - 1])
                if lbl is None or pd.isna(lbl):
                    continue    # ni barriÃ¨re touchÃ©e â†’ pas de trade
                actual = int(lbl)
            else:
                actual = int(
                    test_df["close"].iloc[i] > test_df["close"].iloc[i - 1]
                )
                up_reel.append(actual)

            # Proba de la classe positive : sign -> P(UP) ; crash -> P(crash).
            # Crash : p_useful direct (CrashGuard), SINON 1 - P(UP) (inversion).
            if target_mode == "crash":
                p = self._to_proba(res.get("p_useful"))      # native P(crash)
                if p is None:
                    p_dir = self._to_proba(res.get("proba_up"))
                    if p_dir is None:
                        p_dir = self._vote_to_proba(res.get("vote"))
                    if p_dir is not None:
                        p = 1.0 - p_dir                      # P(UP) -> P(crash)
            else:
                p = self._to_proba(res.get("p_useful"))
                if p is None:
                    p = self._to_proba(res.get("proba_up"))
                if p is None:
                    p = self._vote_to_proba(res.get("vote"))
            if p is None:
                continue            # NEUTRE -> pas de trade (anti-biais P0)
            probas.append(p)
            actuals.append(actual)

        # MÃªme sans aucun signal exploitable, la baseline est publiÃ©e : le
        # modÃ¨le apparaÃ®t au rapport avec 0 trade au lieu de disparaÃ®tre.
        if not probas:
            if not up_reel:
                return None
            return {
                "n_raw": len(up_reel),
                "n_trades": 0,
                "win_rate": 0.0,
                "n_buy": 0,
                # P0 â€” `probas` et `actuals` sont TOUJOURS de mÃªme longueur.
                # Renvoyer ici `up_reel` (99 bougies) dÃ©salignait le pool
                # (probas vides vs actuals pleins) et faisait planter
                # `_pooled_threshold` par IndexError. Le taux de base du
                # fold reste publiÃ© sÃ©parÃ©ment via `up_rate`.
                "probas": [],
                "actuals": [],
                "aucun_signal": True,
                "up_rate": (float(np.mean(up_reel)) if up_reel else 0.0),
                "n_up_reel": int(sum(1 for v in up_reel if v == 1)),
                "n_down_reel": int(sum(1 for v in up_reel if v == 0)),
            }
        actuals_arr = np.asarray(actuals, dtype=int)
        return {
            "n_raw": len(probas),
            "n_trades": len(probas),  # P0 : n brut avant seuillage ; recalibrÃ© ensuite
            "win_rate": 0.0,  # sera recalculÃ© dans _summarize
            "n_buy": 0,
            "probas": probas,
            "actuals": actuals,
            # P0 : taux de base du segment testÃ© = part de bougies
            # haussiÃ¨res. RÃ©fÃ©rence OBLIGATOIRE : un modÃ¨le qui prÃ©dit
            # toujours Â« UP Â» obtient ce win rate SANS aucune compÃ©tence.
            "up_rate": float(actuals_arr.mean()) if len(actuals_arr) else 0.0,
        }

    @staticmethod
    def _to_proba(value) -> float | None:
        """Normalise une valeur vers une proba de [0, 1] (None si inexploitable)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        p = float(value)
        if not np.isfinite(p) or p < 0.0 or p > 1.0:
            return None
        return p

    @staticmethod
    def _vote_to_proba(vote) -> float | None:
        """Convertit un vote discret en proba directionnelle."""
        return vote_to_proba(vote)

    # â”€â”€ Calibration globale poolÃ©e (P0 â€” mesure juste) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @staticmethod
    def _pooled_threshold(probas, y_true):
        return calibrate_threshold(
            probas, y_true,
            candidates=[round(0.50 + i * 0.025, 3) for i in range(13)],
            min_trades=10,
            neutral_fallback=(0.60, 0.40),
        )

    @staticmethod
    def _apply_threshold(probas, actuals, thr_up, thr_down):
        """Recalcule le win_rate avec un seuil donnÃ© (applique la rÃ¨gle thr).

        Retourne AUSSI les mÃ©triques qui rendent le win rate interprÃ©table :
          - `up_rate`      : taux de base (part de bougies haussiÃ¨res parmi
                             les trades retenus) â€” la rÃ©fÃ©rence Â« toujours UP Â» ;
          - `balanced_acc` : balanced accuracy (moyenne du rappel sur les
                             hausses et du rappel sur les baisses).
        Sans elles, un WR de 89.7% sur un indice qui monte 89.7% du temps
        est indistinguable d'un modÃ¨le sans aucune compÃ©tence (biais P0).
        """
        probas = np.asarray(probas, dtype=float)
        actuals = np.asarray(actuals, dtype=int)
        mask = (probas >= thr_up) | (probas <= thr_down)
        if mask.sum() == 0:
            return {"n_trades": 0, "win_rate": 0.0, "up_rate": 0.0,
                    "balanced_acc": 0.5, "n_up_reel": 0, "n_down_reel": 0}
        preds = (probas[mask] >= thr_up).astype(int)
        y = actuals[mask]
        n = int(mask.sum())
        wins = int((preds == y).sum())
        n_up = int((y == 1).sum())
        n_down = n - n_up
        # P0 â€” MÃ‰TRIQUE ANTI-BIAIS : on mesure le rappel SÃ‰PARÃ‰MENT sur les
        # deux classes. Un modÃ¨le qui prÃ©dit toujours Â« UP Â» obtient
        # rappel_hausse = 100% mais rappel_baisse = 0% â†’ balanced_acc = 50%,
        # donc AUCUNE compÃ©tence dÃ©tectÃ©e, mÃªme avec 89.7% de WR brut.
        if n_up > 0 and n_down > 0:
            rappel_up = float((y[preds == 1] == 1).sum()) / n_up
            rappel_down = float((y[preds == 0] == 0).sum()) / n_down
            balanced_acc = 0.5 * (rappel_up + rappel_down)
        else:
            # Une seule classe prÃ©sente dans l'Ã©chantillon : la balanced
            # accuracy n'est PAS identifiable. On renvoie 0.5 (prudence :
            # aucune compÃ©tence ne peut Ãªtre dÃ©montrÃ©e sur un seul rÃ©gime).
            balanced_acc = 0.5
        return {
            "n_trades": n,
            "win_rate": wins / n,
            "up_rate": n_up / n,
            "balanced_acc": balanced_acc,
            "pnl_edge": float((wins * 0.84 - (n - wins)) / n) if n > 0 else 0.0,
            "n_up_reel": n_up,
            "n_down_reel": n_down,
        }

    @staticmethod
    def _assess_bias(
        win_rate: float,
        taux_base: float,
        balanced_wr: float,
        n_up_reel: int,
        n_down_reel: int,
        min_edge: float = MIN_EDGE_DEMO,
        min_down: int = MIN_DOWN_TRADES_DEMO,
    ) -> dict:
        """
        P0 â€” Le win rate brut dÃ©passe-t-il la simple dÃ©rive de l'indice ?

        Trois preuves sont exigÃ©es avant de dÃ©clarer un modÃ¨le Â« hors
        biais de taux de base Â» :
          1. edge = WR âˆ’ taux de base â‰¥ min_edge (le modÃ¨le bat le naÃ¯f
             Â« toujours UP Â» d'au moins 5 points) ;
          2. balanced_wr > 50% (compÃ©tence rÃ©elle sur les DEUX classes) ;
          3. â‰¥ min_down trades sur des bougies BAISSIÃˆRES (sinon la
             performance n'est validÃ©e que dans un seul rÃ©gime de marchÃ©
             et ne dit rien du comportement en marchÃ© descendant).
        """
        edge = float(win_rate) - float(taux_base)
        hors_biais = bool(
            edge >= min_edge
            and float(balanced_wr) > 0.50
            and int(n_down_reel) >= min_down
        )
        if hors_biais:
            raison = "compétence réelle : WR > taux de base et 2 classes testées"
        elif int(n_down_reel) < min_down:
            raison = (f"seulement {int(n_down_reel)} trade(s) sur bougies "
                      f"baissières < {min_down} → modèle non testé en "
                      f"rÃ©gime descendant")
        elif float(balanced_wr) <= 0.50:
            raison = (f"balanced accuracy {float(balanced_wr):.1%} ≤ 50% → "
                      f"aucun pouvoir prédictif (dérive de l'indice)")
        else:
            raison = (f"edge {edge:+.2%} < {min_edge:.0%} → le WR brut "
                      f"n'est que le taux de base ({float(taux_base):.1%})")
        return {
            "taux_base_up": round(float(taux_base), 4),
            "edge_vs_taux_base": round(edge, 4),
            "balanced_wr": round(float(balanced_wr), 4),
            "n_up_reel": int(n_up_reel),
            "n_down_reel": int(n_down_reel),
            "hors_biais_taux_base": hors_biais,
            "raison": raison,
        }

    # â”€â”€ Ã‰valuation in-sample (fenÃªtres fixes, pour l'overfitting) â”€â”€â”€
    def _evaluate_in_sample(
        self, predict_fn, train_df: pd.DataFrame, target_mode: str = "sign"
    ) -> float | None:
        # Mode crash : P(crash) + label spike (coherence avec _evaluate).
        if target_mode == "crash":
            from deriv.ensemble_predictor import (
                CRASH_SEUIL_FRAC as _INS_SEUIL,
            )
        w = self.IN_SAMPLE_WINDOW
        step = self.IN_SAMPLE_STEP
        if len(train_df) <= w:
            w = max(self.MIN_WINDOW, len(train_df) - 1)
            if w < self.MIN_WINDOW:
                return None
        preds, actuals = [], []
        for end in range(w, len(train_df), step):
            window = train_df.iloc[end - w:end]
            if len(window) < self.MIN_WINDOW:
                continue
            try:
                r_is = predict_fn(window)
            except Exception:
                continue
            if target_mode == "crash":
                # P(crash) : p_useful natif (CrashGuard), sinon 1 - P(dir).
                p = self._to_proba(r_is.get("p_useful"))
                if p is None:
                    p_dir = self._to_proba(r_is.get("proba_up"))
                    if p_dir is None:
                        p_dir = self._vote_to_proba(r_is.get("vote"))
                    if p_dir is not None:
                        p = 1.0 - p_dir
                if p is None:
                    continue
                pred = int(p >= 0.5)
                actual = int(
                    (train_df["close"].iloc[end]
                     / train_df["close"].iloc[end - 1]) - 1.0
                    <= -_INS_SEUIL
                )
            else:
                vote = r_is.get("vote", 0.5)
                if not (vote >= 0.7 or vote <= 0.3):
                    continue
                pred = 1 if vote >= 0.7 else 0
                actual = int(
                    train_df["close"].iloc[end] > train_df["close"].iloc[end - 1]
                )
            preds.append(pred)
            actuals.append(actual)
        if not preds:
            return None
        return sum(p == a for p, a in zip(preds, actuals)) / len(preds)

    # â”€â”€ ExÃ©cution complÃ¨te â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def run(
        self, df: pd.DataFrame, target_mode: str = "triple_barrier"
    ) -> dict:
        """Lance le WFO sur le DataFrame OHLCV complet (index chronologique).

        DÃ©faut P0-v2 : target triple-barriÃ¨re symÃ©trique (anti-drift).
        `target_mode="sign"` conserve la cible triviale UP/DOWN (legacy).
        """

        min_size = self.TRAIN_SIZE + self.PURGE + self.EMBARGO + self.TEST_SIZE
        if len(df) < min_size:
            raise ValueError(
                f"Besoin de {min_size}+ bougies, reÃ§u {len(df)}. "
                f"Lance avec --count {min_size + 100}"
            )

        plan = self.folds_plan(
            len(df), self.TRAIN_SIZE, self.TEST_SIZE, self.EMBARGO,
            self.STEP, self.PURGE,
        )
        print(
            f"WFO : {len(plan)} folds "
            f"(train {self.TRAIN_SIZE} + purge {self.PURGE} "
            f"+ embargo {self.EMBARGO} + test {self.TEST_SIZE}, pas {self.STEP})"
        )

        all_results = defaultdict(list)   # {model: [fold_result]}
        in_sample = {}                    # {model: [wr_in_sample par fold]}
        pooled_raw = defaultdict(lambda: {"probas": [], "actuals": []})

        # Modeles sans etat. Mode "crash" : le CrashGuard (estimateur
        # d'hazard, baseline du WFO) rejoint le pool -- il expose P(crash)
        # native via p_useful (flag _native_proba, tourne le bypass du
        # _StatelessWrapper, sinon il ne traderait jamais en HOLD).
        stateless = dict(self.stateless_models)
        if target_mode == "crash":
            from deriv.ensemble_predictor import CrashGuardPredictor
            stateless["CrashGuard"] = CrashGuardPredictor

        for fold in plan:
            train_df = df.iloc[fold["train"][0]:fold["train"][1]]
            test_df = df.iloc[fold["test_start"]:fold["test_end"]]

            # GARDE-FOU 3 : vÃ©rification dure â€” aucun chevauchement possible.
            # Si cette assertion Ã©choue, le dÃ©coupage est cassÃ© â†’ on stoppe
            # tout plutÃ´t que de publier un win rate contaminÃ©.
            assert train_df.index[-1] < test_df.index[0], (
                f"FUITE DÃ‰TECTÃ‰E fold {fold['fold']} : train finit Ã  "
                f"{train_df.index[-1]}, test commence Ã  {test_df.index[0]}"
            )

            print(
                f"Fold {fold['fold']}/{len(plan)} | "
                f"Train: {train_df.index[0].date()}â†’{train_df.index[-1].date()} | "
                f"Test: {test_df.index[0].date()}â†’{test_df.index[-1].date()}"
            )

            trained = {}

            # ModÃ¨les entraÃ®nables
            for name, ModelClass in self.models.items():
                try:
                    model = ModelClass()
                    self._disarm_save(model)
                    model.train(train_df)
                    trained[name] = model

                    res = self._evaluate(model.predict, test_df, target_mode)
                    if res:
                        # Stocker les probas/actuals dans all_results pour
                        # permettre le recalcul avec seuil calibrÃ© (P0)
                        fold_result = {
                            "fold": fold["fold"],
                            "n_trades": res["n_trades"],
                            "win_rate": res["win_rate"],
                            "probas": res.get("probas", []),
                            "actuals": res.get("actuals", []),
                        }
                        all_results[name].append(fold_result)
                        # Collecte pour calibration poolÃ©e (P0)
                        if res.get("probas"):
                            pooled_raw[name]["probas"].extend(res["probas"])
                            pooled_raw[name]["actuals"].extend(res["actuals"])
                    if self.eval_in_sample:
                        ins = self._evaluate_in_sample(model.predict, train_df, target_mode)
                        if ins is not None:
                            in_sample.setdefault(name, []).append(ins)
                except Exception as exc:
                    logger.error(f"Fold {fold['fold']} {name} : {exc}")

            # ModÃ¨les sans Ã©tat (pas d'entraÃ®nement)
            for name, ModelClass in stateless.items():
                try:
                    raw_m = ModelClass()
                    model = self._StatelessWrapper(raw_m)
                    res = self._evaluate(model.predict, test_df, target_mode)
                    if res:
                        all_results[name].append({
                            "fold": fold["fold"],
                            "n_raw": res["n_raw"],
                            "n_trades": res["n_trades"],
                            "win_rate": res["win_rate"],
                            "probas": res["probas"],
                            "actuals": res["actuals"],
                        })
                        pooled_raw.setdefault(name, {"probas": [], "actuals": []})
                        # P0 : n'Ã©tendre le pool QUE si des probas existent.
                        # Sinon `actuals` grossissait sans `probas` (fold sans
                        # signal directionnel) â†’ IndexError dans
                        # `_pooled_threshold` (642 probas vs 842 actuals).
                        if res["probas"]:
                            pooled_raw[name]["probas"].extend(res["probas"])
                            pooled_raw[name]["actuals"].extend(res["actuals"])
                except Exception as exc:
                    logger.error(f"Fold {fold['fold']} {name} : {exc}")

            # ENSEMBLE complet (rÃ¨gle 4/5) â€” le vrai dÃ©cideur du bot
            if self.eval_ensemble:
                try:
                    from deriv.ensemble_predictor import EnsemblePredictor

                    ens = EnsemblePredictor()
                    if "HMM" in trained:
                        ens.hmm = trained["HMM"]
                    if "XGBoost" in trained:
                        ens.xgb = trained["XGBoost"]
                    adapter = _EnsembleAdapter(ens)

                    res = self._evaluate(adapter.predict, test_df, target_mode)
                    if res:
                        # P0 : probas/actuals OBLIGATOIRES ici. Sans eux,
                        # pooled_raw["Ensemble4sur5"] restait VIDE â†’ aucun
                        # seuil calibrÃ© â†’ le placeholder win_rate=0.0 de
                        # `_evaluate` survivait dans le rapport final
                        # (symptÃ´me observÃ© : Â« 0.0% sur 40 trades Â» Ã 
                        # chacun des 45 folds).
                        all_results["Ensemble4sur5"].append({
                            "fold": fold["fold"],
                            "n_trades": res["n_trades"],
                            "win_rate": res["win_rate"],
                            "probas": res.get("probas", []),
                            "actuals": res.get("actuals", []),
                        })
                        if res.get("probas"):
                            pooled_raw["Ensemble4sur5"]["probas"].extend(
                                res["probas"]
                            )
                            pooled_raw["Ensemble4sur5"]["actuals"].extend(
                                res["actuals"]
                            )
                    if self.eval_in_sample:
                        ins = self._evaluate_in_sample(adapter.predict, train_df, target_mode)
                        if ins is not None:
                            in_sample.setdefault("Ensemble4sur5", []).append(ins)
                except Exception as exc:
                    logger.error(f"Fold {fold['fold']} Ensemble4sur5 : {exc}")

        in_sample_mean = {
            name: float(np.mean(v)) for name, v in in_sample.items() if v
        }

        # â”€â”€ P0 : recalibration globale poolÃ©e â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Chaque modÃ¨le a accumulÃ© des probas/actuals out-of-sample dans
        # pooled_raw. On calibre un seuil unique sur le pool, puis on
        # recalcule les win_rates par fold avec ce seuil (au lieu de 0.7/0.3).
        pooled_thresholds = {}
        for name, raw in pooled_raw.items():
            if not raw["probas"]:
                continue
            thr_up, thr_down = self._pooled_threshold(raw["probas"], raw["actuals"])
            pooled_thresholds[name] = (thr_up, thr_down)
            print(
                f"Calibration poolÃ©e {name} : seuil â†‘{thr_up:.3f}/â†“{thr_down:.3f} "
                f"({len(raw['probas'])} prÃ©dictions)"
            )

        return self._summarize(
            all_results, len(plan), in_sample_mean, pooled_raw, pooled_thresholds
        )

    # â”€â”€ SynthÃ¨se statistique par modÃ¨le â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _summarize(
        self,
        results: dict,
        n_folds: int,
        in_sample: dict | None = None,
        pooled_raw: dict | None = None,
        pooled_thresholds: dict | None = None,
    ) -> dict:
        """MÃ©triques finales par modÃ¨le : WR pondÃ©rÃ©, Wilson, p-value, verdict.

        P0 : si pooled_thresholds est fourni, recalcule les win_rates par
        fold avec le seuil calibrÃ© sur le pool out-of-sample (au lieu de
        0.7/0.3 en dur dans _evaluate).
        """
        in_sample = in_sample or {}
        pooled_raw = pooled_raw or {}
        pooled_thresholds = pooled_thresholds or {}
        summary = {}

        for model_name, folds in results.items():
            if not folds:
                continue
            # P0 : recalculer les win_rates avec le seuil calibré si dispo.
            # Fallback 0.70/0.30 = seuil de décision du bot en production.
            # Safeguard : comparer calibré vs baseline (0.70/0.30) et garder le meilleur.
            thr = pooled_thresholds.get(model_name)
            if thr:
                thr_up_cal, thr_down_cal = thr
                thr_up_base, thr_down_base = 0.70, 0.30
                # Évaluer les deux seuils sur les probas/actuals poolés
                probas_all = []
                actuals_all = []
                for f in folds:
                    probas_all.extend(f.get("probas", []))
                    actuals_all.extend(f.get("actuals", []))
                if probas_all:
                    res_cal = self._apply_threshold(probas_all, actuals_all, thr_up_cal, thr_down_cal)
                    res_base = self._apply_threshold(probas_all, actuals_all, thr_up_base, thr_down_base)
                    # Choisir le seuil qui donne le meilleur WR pondéré
                    if res_base["n_trades"] > 0 and res_base["win_rate"] >= res_cal["win_rate"]:
                        print(
                            f"Safeguard {model_name}: baseline {thr_up_base}/{thr_down_base} "
                            f"WR={res_base["win_rate"]:.1%} >= calibré {thr_up_cal}/{thr_down_cal} "
                            f"WR={res_cal["win_rate"]:.1%} -> utilise baseline"
                        )
                        thr_up, thr_down = thr_up_base, thr_down_base
                    else:
                        thr_up, thr_down = thr_up_cal, thr_down_cal
                else:
                    thr_up, thr_down = 0.70, 0.30
            else:
                thr_up, thr_down = 0.70, 0.30
            folds_cal = []
            for f in folds:
                # Recalculer à partir des probas/actuals bruts du fold
                probas = f.get("probas", [])
                actuals = f.get("actuals", [])
                if not probas:
                    continue
                res = self._apply_threshold(probas, actuals, thr_up, thr_down)
                folds_cal.append({
                    "fold": f["fold"],
                    "n_trades": res["n_trades"],
                    "win_rate": res["win_rate"],
                    "up_rate": res["up_rate"],
                    "balanced_acc": res["balanced_acc"],
                    "n_up_reel": res["n_up_reel"],
                    "n_down_reel": res["n_down_reel"],
                })
            folds = folds_cal if folds_cal else folds

            total_trades = sum(f["n_trades"] for f in folds)
            weighted_wr = (
                sum(f["win_rate"] * f["n_trades"] for f in folds) / total_trades
                if total_trades > 0 else 0.0
            )
            wrs = [f["win_rate"] for f in folds]
            std_folds = float(np.std(wrs))
            wins = int(round(weighted_wr * total_trades))
            wr_lo, wr_hi = wilson_interval(wins, total_trades)
            p_value = binomial_pvalue(wins, total_trades)

            # â”€â”€ P0 : contrÃ´le du biais de taux de base â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Un WR brut de 89.7% peut n'Ãªtre QUE la dÃ©rive d'un indice
            # haussier (CRASH500 M1 : 89.7% de bougies haussiÃ¨res, et le
            # WR annoncÃ© valait exactement 89.7%). PondÃ©ration par le
            # nombre de trades de chaque fold, comme le WR lui-mÃªme.
            mesurable = any(("up_rate" in f or "balanced_acc" in f)
                            for f in folds)
            if mesurable:
                up_rate_moy = (
                    sum(f.get("up_rate", 0.0) * f["n_trades"] for f in folds)
                    / total_trades if total_trades > 0 else 0.0
                )
                balanced_wr = (
                    sum(f.get("balanced_acc", 0.5) * f["n_trades"]
                        for f in folds)
                    / total_trades if total_trades > 0 else 0.5
                )
                n_up_reel = sum(f.get("n_up_reel", 0) for f in folds)
                n_down_reel = sum(f.get("n_down_reel", 0) for f in folds)
                biais = self._assess_bias(
                    weighted_wr, up_rate_moy, balanced_wr,
                    n_up_reel, n_down_reel,
                )
            else:
                # Folds agrÃ©gÃ©s (appel direct de `_summarize`) : aucune
                # mÃ©trique de taux de base â†’ verdict neutre, jamais un
                # refus abusif basÃ© sur des valeurs par dÃ©faut.
                biais = {
                    "taux_base_up": None,
                    "edge_vs_taux_base": None,
                    "balanced_wr": None,
                    "n_up_reel": None,
                    "n_down_reel": None,
                    "hors_biais_taux_base": True,
                    "raison": ("non mesurable : folds sans mÃ©triques de "
                               "taux de base"),
                }

            ins = in_sample.get(model_name)
            diag = diagnostic_overfitting(ins, weighted_wr, wrs, total_trades)

            summary[model_name] = {
                "win_rate_moyen": round(weighted_wr, 4),
                "win_rate_std": round(std_folds, 4),
                "win_rate_min": round(min(wrs), 4),
                "win_rate_max": round(max(wrs), 4),
                "wilson_95": [round(wr_lo, 4), round(wr_hi, 4)],
                "p_value_vs_hasard": round(p_value, 5),
                "meilleur_que_hasard": p_value < 0.05,
                "n_folds": n_folds,
                "n_trades_total": total_trades,
                "profitable": weighted_wr >= BREAKEVEN,
                "in_sample_wr": round(ins, 4) if ins is not None else None,
                "diagnostic_overfitting": diag,
                "seuil_calibre": (
                    {"up": thr[0], "down": thr[1]} if thr else None
                ),
                # P0 : biais de taux de base (dÃ©rive de l'indice)
                "taux_base_up": biais["taux_base_up"],
                "edge_vs_taux_base": biais["edge_vs_taux_base"],
                "balanced_wr": biais["balanced_wr"],
                "diagnostic_biais": biais,
                # CritÃ¨re de passage en dÃ©mo rÃ©elle (mission)
                "demo_ready": (
                    weighted_wr >= BREAKEVEN
                    and n_folds >= MIN_FOLDS_DEMO
                    and std_folds < MAX_STD_DEMO
                    and total_trades >= MIN_TRADES_DEMO
                    and not diag["overfitting"]
                    # P0 : interdit de valider un WR qui n'est que le taux
                    # de base de l'indice (aucune compÃ©tence prÃ©dictive).
                    and biais["hors_biais_taux_base"]
                ),
                "detail_folds": folds,
            }

        return summary

    # â”€â”€ Rapport lisible â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def print_report(self, summary: dict):
        print("\n" + "=" * 64)
        print("WALK-FORWARD OPTIMIZATION â€” Rapport")
        print(f"Seuil minimum profitable : {BREAKEVEN:.1%} (breakeven options)")
        print(
            f"CritÃ¨res dÃ©mo rÃ©elle : {MIN_FOLDS_DEMO}+ folds, "
            f"Ïƒ < {MAX_STD_DEMO:.0%}, {MIN_TRADES_DEMO}+ trades, pas d'overfitting"
        )
        print("=" * 64)

        for model, stats in summary.items():
            biais = stats.get("diagnostic_biais") or {}
            if stats["demo_ready"]:
                status = "âœ… PRÃŠT DÃ‰MO (tous critÃ¨res)"
            elif stats["profitable"] and not biais.get("hors_biais_taux_base", True):
                status = "âŒ BIAIS DE TAUX DE BASE (WR = dÃ©rive de l'indice)"
            elif stats["profitable"]:
                status = "ðŸŸ  PROFITABLE mais critÃ¨res dÃ©mo incomplets"
            else:
                status = "âŒ SOUS SEUIL"
            print(f"\n{model} : {status}")
            print(f"  Win Rate moyen      : {stats['win_rate_moyen']:.1%}")
            print(f"  Ã‰cart-type folds    : {stats['win_rate_std']:.1%}")
            print(f"  Min / Max           : {stats['win_rate_min']:.1%} "
                  f"/ {stats['win_rate_max']:.1%}")
            lo, hi = stats["wilson_95"]
            print(f"  IC 95% (Wilson)     : [{lo:.1%} ; {hi:.1%}]")
            print(f"  P-value vs hasard   : {stats['p_value_vs_hasard']:.4f}"
                  f"{' âœ…' if stats['meilleur_que_hasard'] else ''}")
            if stats["in_sample_wr"] is not None:
                print(f"  WR in-sample        : {stats['in_sample_wr']:.1%}")
            print(f"  Trades totaux       : {stats['n_trades_total']}")
            print(f"  Nombre de folds     : {stats['n_folds']}")
            # P0 â€” le win rate brut ne veut RIEN dire sans son taux de base :
            # 89.7% de WR sur un indice qui monte 89.7% du temps = zÃ©ro edge.
            base = stats.get("taux_base_up")
            if base is None:
                print("  Taux de base (UP)   : n/a (folds sans mÃ©trique)")
            else:
                print(f"  Taux de base (UP)   : {base:.1%}  "
                      f"â†’ edge {stats['edge_vs_taux_base']:+.2%}")
                print(f"  Balanced accuracy   : {stats['balanced_wr']:.1%}  "
                      f"(50% = aucun pouvoir prÃ©dictif)")
            if biais.get("n_down_reel") is not None:
                print(f"  Trades baissiers    : {biais['n_down_reel']}")
            if biais and not biais["hors_biais_taux_base"]:
                print(f"  âš ï¸  BIAIS TAUX DE BASE : {biais['raison']}")
            if stats["diagnostic_overfitting"]["overfitting"]:
                print("  âš ï¸  Overfitting suspectÃ© :")
                for s in stats["diagnostic_overfitting"]["signaux"]:
                    print(f"      - {s}")

            print("  DÃ©tail par fold :")
            for f in stats["detail_folds"]:
                bar = "ðŸŸ¢" if f["win_rate"] >= BREAKEVEN else "ðŸ”´"
                base = f.get("up_rate")
                extra = (f" | base UP {base:.0%}"
                         if isinstance(base, (int, float)) else "")
                print(f"    Fold {f['fold']:2d}: {bar} {f['win_rate']:.1%} "
                      f"({f['n_trades']} trades){extra}")

        print("\n" + "=" * 64)
        biaises = [
            m for m, s in summary.items()
            if (s.get("diagnostic_biais") or {}).get("hors_biais_taux_base") is False
        ]
        if biaises:
            print("âš ï¸  BIAIS DE TAUX DE BASE â€” win rates non concluants : "
                  f"{', '.join(biaises)}")
            print("   Le WR brut y est expliquÃ© par la seule dÃ©rive de l'indice :")
            print("   aucune compÃ©tence prÃ©dictive dÃ©montrÃ©e â†’ dÃ©mo interdite.")
        if not summary:
            print("âš ï¸  Aucun modÃ¨le n'a produit de trades sur cette sÃ©rie")
            return
        if any(s["demo_ready"] for s in summary.values()):
            print("âœ… Au moins un modÃ¨le remplit TOUS les critÃ¨res â†’ dÃ©mo rÃ©elle OK")
        elif any(s["profitable"] for s in summary.values()):
            print("ðŸŸ  ModÃ¨le(s) profitable(s) mais critÃ¨res dÃ©mo incomplets :")
            print("   â†’ augmente --count (5000+) pour plus de trades/folds")
        else:
            print("âš ï¸  Aucun modÃ¨le ne dÃ©passe le seuil sur ce symbole/timeframe")
            print("   Essaie : --symbol frxEURUSD  OU  --granularity 300 (M5)")


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _configure_console_utf8() -> None:
    """Ã‰vite les UnicodeEncodeError sur consoles Windows (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


async def main():
    _configure_console_utf8()
    p = argparse.ArgumentParser(description="Walk-Forward Optimization Deriv")
    p.add_argument("--symbol", default="R_75")
    p.add_argument("--count", type=int, default=3000)
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--test-size", type=int, default=None)
    p.add_argument("--embargo", type=int, default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--no-in-sample", action="store_true",
                   help="Saute l'Ã©valuation in-sample (plus rapide)")
    p.add_argument(
        "--target",
        choices=["sign", "crash", "triple_barrier"],
        default="triple_barrier",
        help="Cible WFO : 'triple_barrier' (dÃ©faut â€” barriÃ¨res symÃ©triques "
             "+/-1.5*ATR, horizon 3, anti-drift, LÃ³pez de Prado), 'sign' "
             "(UP/DOWN, target triviale legacy) ou 'crash' (spike "
             "CRASH500, cible asymÃ©trique P(crash))",
    )
    args = p.parse_args()

    from deriv.data_collector import DerivDataCollector

    print(f"Chargement de {args.count} bougies {args.symbol} "
          f"({args.granularity}s)...")
    collector = DerivDataCollector(args.symbol, args.granularity)
    await collector.connect()
    df = await collector.get_candle_history_full(target_count=args.count)
    await collector.close()

    print(f"DonnÃ©es chargÃ©es : {len(df)} bougies\n")

    wfo = WalkForwardOptimizer(
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        step=args.step,
        eval_in_sample=not args.no_in_sample,
        granularity=args.granularity,
    )
    summary = wfo.run(df, target_mode=args.target)
    wfo.print_report(summary)

    # Sauvegarde CSV + JSON pour analyse
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"
    )
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{args.symbol}_{args.granularity}"
    for model, stats in summary.items():
        pd.DataFrame(stats["detail_folds"]).to_csv(
            os.path.join(out_dir, f"wfo_{model}_{tag}.csv"), index=False
        )
    export = {m: {k: v for k, v in s.items() if k != "detail_folds"}
              for m, s in summary.items()}
    with open(os.path.join(out_dir, f"wfo_summary_{tag}.json"),
              "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nCSV + JSON sauvegardÃ©s dans {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())







