"""
═══════════════════════════════════════════════════════════════════
ÉTAPE 2 — CROISEMENT DE TOUS LES ALGORITHMES
deriv/ensemble_predictor.py
═══════════════════════════════════════════════════════════════════

Croisement professionnel de 5 modèles :
1. HMM (Baum-Welch + Forward-Backward + Viterbi)
2. XGBoost (features tabulaires)
3. LSTM (mémoire temporelle longue) — optionnel (nécessite tensorflow)
4. Filtre de Kalman (momentum)
5. RSI Divergence (retournements)

Vote à la majorité QUALIFIÉE : 4/5 minimum pour déclencher un trade.
En dessous de 4/5 → HOLD. Toujours. (RÈGLE ABSOLUE)
"""
from __future__ import annotations

import logging
import os
import pickle
from deriv.constants import safe_pickle_load
from deriv.voting import proba_to_vote
from deriv.threshold_calibrator import calibrate_threshold

import numpy as np
import pandas as pd
import ta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

# ── Dépendances optionnelles (dégradation propre si absentes) ─────────
try:
    from xgboost import XGBClassifier
    import xgboost as _xgb
    _XGB_OK = True
    _XGB_ERROR = ""
except Exception as exc:  # pragma: no cover
    _XGB_OK = False
    _XGB_ERROR = repr(exc)
    XGBClassifier = None

try:
    import tensorflow as tf  # noqa: F401
    _TF_OK = True
    _TF_ERROR = ""
except Exception as exc:  # pragma: no cover
    _TF_OK = False
    _TF_ERROR = repr(exc)
    tf = None

# Chemins de modèles ancrés au module (pas au CWD — robuste)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_BASE_DIR, "models")


def _model_path(name: str) -> str:
    os.makedirs(MODELS_DIR, exist_ok=True)
    return os.path.join(MODELS_DIR, name)


# ── CIBLE CRASH (P1 — pari asymétrique CRASH500) ────────────────────
# Seuil CALIBRÉ empiriquement par measure_crash.py → crash_stats.txt :
#   seuil retenu (p2) : -0.1537 %   → ~2 % des bougies M1 = crashs.
# Sur un indice en dérive (89.76 % UP), la cible « UP/DOWN » ne mesure
# que la dérive ; la cible « crash » (ret <= -seuil) isole l'événement
# rare & tradeable. Le modèle prédit P(crash) ; le WR sera bas (~2 %),
# mais l'edge (rappel / précision) devient exploitable en short / options.
CRASH_SEUIL_FRAC = 0.001537   # = 0.1537 % (p2 négatif CRASH500 M1)


# ── FEATURE ENGINEERING ─────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit 8 features techniques stables pour XGBoost et LSTM.

    ⚠️  ANTI-FUITE : `target` est ABSENT de ce DataFrame.
    Utiliser `build_labels(df)` séparément pour générer les labels
    d'entraînement. Ce découplage empêche toute fuite de données futures
    dans les features (López de Prado, 2018).

    ⚠️  ANTI-SURAPPRENTISSAGE : 8 features sélectionnées empiriquement
    (stabilité inter-folds WFO H4) au lieu de 15. Moins de dimensions
    = moins d'overfitting sur les petits datasets H4/H1.
    """
    if df is None or df.empty:
        # Données absentes → DataFrame vide (dégradation propre)
        return pd.DataFrame()
    d = df.copy()
    # ⚠️  ffill uniquement sur le passé (forward fill = passé vers futur)
    #     Pas de bfill pour ne pas propager le futur dans le passé.
    d = d.ffill()
    if not {"open", "high", "low", "close"}.issubset(d.columns):
        # Colonnes OHLC incomplètes → DataFrame vide
        return pd.DataFrame()
    if len(d) < 60:
        # Trop court pour EMA50/BB/ADX → indicateurs tous NaN → vide
        return pd.DataFrame()
    c = d["close"]
    h = d["high"]
    l = d["low"]

    # ── 8 features stables (sélectionnées par stabilité inter-folds) ──
    d["rsi_14"]    = ta.momentum.RSIIndicator(c, 14).rsi()
    d["roc_10"]    = ta.momentum.ROCIndicator(c, 10).roc()
    d["macd_diff"] = ta.trend.MACD(c).macd_diff()
    d["adx_14"]    = ta.trend.ADXIndicator(h, l, c, 14).adx()
    bb = ta.volatility.BollingerBands(c, 20, 2)
    d["bb_pct"]    = bb.bollinger_pband()   # position dans les bandes (0-1)
    d["atr_14"]    = ta.volatility.AverageTrueRange(h, l, c, 14).average_true_range()
    d["log_ret"]   = np.log(c / c.shift(1))
    d["log_ret_5"] = np.log(c / c.shift(5))

    # ── P2 : Features temporelles (saisonnalité forex) ──────────────
    # Le forex H4 a une saisonnalité connue (overlap Londres/NY, asiatique).
    # Ces features sont déterministes (pas de fuite temporelle) et aident
    # le modèle à apprendre les sessions favorables.
    idx = d.index
    if hasattr(idx, "hour"):
        hour = idx.hour
        d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        d["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        d["day_of_week"] = idx.dayofweek.astype(float) / 6.0  # 0=lundi, 1=dim

        # ── V2 : SESSIONS (binaire) — recherche 2025 : London-NY overlap =
        # meilleure liquidité → signaux plus fiables.
        hours = idx.hour + idx.minute / 60.0
        d["london_active"]  = ((hours >= 7) & (hours < 16)).astype(float)
        d["newyork_active"] = ((hours >= 13) & (hours < 21)).astype(float)
        d["overlap_active"] = ((hours >= 13) & (hours < 16)).astype(float)
        d["asia_active"]    = (((hours >= 21) | (hours < 7))).astype(float)

    if hasattr(idx, "dayofweek"):
        dow = idx.dayofweek
        d["is_monday"] = (dow == 0).astype(float)   # lundi = gaps fréquents
        d["is_friday"] = (dow == 4).astype(float)   # vendredi = clôture précoce

    # ── V2 : LIQUIDITÉ (proxy volume) ────────────────────────────────
    # Pas de carnet d'ordres chez Deriv → le volume relatif est le proxy
    # standard (stefan-jansen, 3e éd., famille "liquidité").
    if "volume" in d.columns and d["volume"].notna().any():
        vol = d["volume"].astype(float)
        vol_ma20 = vol.rolling(20).mean()
        d["volume_ratio_5"] = vol / vol_ma20.replace(0, np.nan)
        d["volume_spike"] = (d["volume_ratio_5"] > 2.0).astype(float)
    else:
        # Données sans volume (certains flux Deriv) → constantes neutres
        d["volume_ratio_5"] = 1.0
        d["volume_spike"] = 0.0

    # ── V2 : MICROSTRUCTURE (anatomie de la bougie) ─────────────────
    rng = (h - l).replace(0, np.nan)
    d["body_ratio"]   = ((c - d["open"]).abs() / rng).fillna(0.0)
    d["upper_shadow"] = ((h - d[["close", "open"]].max(axis=1)) / rng).fillna(0.0)
    d["lower_shadow"] = ((d[["close", "open"]].min(axis=1) - l) / rng).fillna(0.0)
    d["bull_candle"]  = (c > d["open"]).astype(float)

    # ── V2 : RÉGIME DE VOLATILITÉ (ATR relatif au prix) ─────────────
    d["atr_pct"] = (d["atr_14"] / c).replace([np.inf, -np.inf], np.nan)

    return d.dropna()


def build_labels(
    df: pd.DataFrame,
    *,
            mode: str = "sign",
    atr_period: int = 14,
    touch_mult: float = 1.5,
    horizon: int = 3,
    crash_seuil: float = CRASH_SEUIL_FRAC,
) -> "pd.Series | None":
    """
    Génère les labels binaires d'entraînement SÉPARÉMENT des features.

    Deux modes (P1 — choix du pari) :

    mode="sign" (défaut, compatible) :
        label[t] = 1 si close[t+1] > close[t], 0 sinon.
        → compatible avec l'ancien code / les tests existants.

    mode="triple_barrier" (López de Prado, 2018) :
        Label = 1 si le prix touche +touch_mult*ATR avant -touch_mult*ATR
        dans les `horizon` bougies suivantes.
        Label = 0 si l'inverse.
        → Ne parie que sur les mouvements DÉCISIFS, ignore le bruit.
                → C'est ce qui transforme 51% sur tout en 57%+ sur le sous-ensemble tradé.

    mode="crash" (cible asymétrique CRASH500) :
        label[t] = 1 si le rendement M1 t→t+1 <= -crash_seuil (bougie de
        crash, ~2 %), 0 sinon. Calibré empiriquement (measure_crash.py) :
        prédire le crash, pas la direction → évite de mesurer la dérive de
        l'indice haussière (89.76 % UP = zéro edge).

    La dernière ligne est toujours exclue (futur inconnu).
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    c = df["close"]

    if mode == "sign":
        labels = (c.shift(-1) > c).astype(int)
        return labels.iloc[:-1]

    if mode == "crash":
        # label[t] = 1 si le prochain rendement franchit le seuil de crash.
        futur_ret = (c.shift(-1) / c) - 1.0
        labels = (futur_ret <= -crash_seuil).astype(int)
        return labels.iloc[:-1]

    if mode == "triple_barrier":
        # ATR pour les barrières
        h, l = df["high"], df["low"]
        atr = ta.volatility.AverageTrueRange(h, l, c, atr_period).average_true_range()
        labels = pd.Series(index=c.index, dtype="Int64")
        for i in range(len(c) - 1):
            if i + horizon >= len(c):
                break
            a = float(atr.iloc[i])
            if a <= 0 or np.isnan(a):
                continue
            upper = float(c.iloc[i]) + touch_mult * a
            lower = float(c.iloc[i]) - touch_mult * a
            touched = 0
            for j in range(i + 1, min(i + 1 + horizon, len(c))):
                if float(h.iloc[j]) >= upper:
                    touched = 1
                    break
                if float(l.iloc[j]) <= lower:
                    touched = -1
                    break
            if touched != 0:
                labels.iloc[i] = 1 if touched == 1 else 0
        return labels.iloc[:-1].dropna()

    raise ValueError(f"mode label inconnu : {mode!r} (attendu 'sign', 'crash' ou 'triple_barrier')")


# ── MODÈLE : CrashGuardPredictor (cible asymétrique CRASH500) ─────────
class CrashGuardPredictor:
    """Détecteur de crash — cible asymétrique CRASH500 M1.

    Mesure empirique (measure_crash.py → crash_stats.txt), 5000 bougies M1 :
      - taux UP global          = 89.76 %  → prédire UP = copier la dérive.
      - seuil crash (p2)        = -0.1537 %  → ~2 % des bougies = crashs.
      - streak AVANT crash      = 0 (med)   → IID : pas de télégraphement.
      - gap entre crashs        = 50 bougies (médiane 34) → Poisson λ=1/50.

    Le CrashGuard est un **estimateur d'hazard calibré** (P(crash) via la
    fonction de survie résiduelle du temps écoulé depuis le dernier spike).
    Il ne « prédit » pas la direction haussière (cela vaudrait copier la
    dérive) : il isole la probabilité d'un événement rare & tradeable
    (short / options), et sert de *baseline* contre laquelle le WFO mesure
    la compétence réelle des autres modèles au seuil de crash.

    Vote : 1.0 (SELL → crash) si P(crash) >= SEUIL_VOTE, sinon 0.5 (HOLD).
    → il ne vote JAMAIS « UP » : « pas de crash » = dérive de l'indice,
    pas un signal d'achat exploitable.

    _native_proba = True : il expose P(crash) via `p_useful` directement,
    sans passer par le normalisateur vote→proba_up du `_StatelessWrapper`
    (sinon son p_useful serait écrasé à None en mode HOLD, perte de signal).
    """
    _native_proba = True          # bypass _StatelessWrapper (proba native)
    CRASH_SEUIL = CRASH_SEUIL_FRAC     # -0.1537 % (p2)
    MEAN_GAP = 50.0                # bougies entre crashs (Poisson λ=1/50)
    BASE_RATE = 0.02               # fréquence empirique des crashs (~2 %)
    SEUIL_VOTE = 0.50              # P(crash) >= 50 % → vote SELL

    def __init__(self, min_bars: int = 30):
        self.MIN_BARS = min_bars

    def predict(self, df: pd.DataFrame) -> dict:
        # Données insuffisantes → HOLD (aucune proba directionnelle).
        if df is None or len(df) < self.MIN_BARS or "close" not in df.columns:
            return {"vote": 0.5, "p_useful": None}
        ret = df["close"].astype(float).pct_change().dropna()
        if ret.empty:
            return {"vote": 0.5, "p_useful": None}
        # Temps écoulé depuis le dernier vrai crash (ret <= -seuil).
        since = 0
        for v in reversed(ret.values):
            if v <= -self.CRASH_SEUIL:
                break
            since += 1
        # Hazard résiduel Poisson : 1 - exp(-t/MEAN_GAP), borné au taux de base.
        p = 1.0 - np.exp(-since / self.MEAN_GAP)
        p = float(min(max(p, self.BASE_RATE), 0.95))
        vote = 1.0 if p >= self.SEUIL_VOTE else 0.5
        return {"vote": vote, "p_useful": p}


def _kalman_filter(prices: pd.Series, gain: float = 0.3) -> pd.Series:
    """Filtre de Kalman simple (prix filtré lisse + gap correction)."""
    filtered = []
    kp, ke = float(prices.iloc[0]), 1.0
    for p in prices:
        kg = gain / (gain + ke)
        kp = kp + kg * (float(p) - kp)
        ke = (1 - kg) * ke
        filtered.append(kp)
    return pd.Series(filtered, index=prices.index)


# FEATURE_COLS : features de production actuelles (XGBoost, LSTM).
FEATURE_COLS = [
    "rsi_14", "roc_10", "macd_diff", "adx_14",
    "bb_pct", "atr_14", "log_ret", "log_ret_5",
    "hour_sin", "hour_cos", "day_of_week",
]

# FEATURE_COLS_V2 : features etendues (sessions, liquidite, microstructure) — futur.
FEATURE_COLS_V2 = FEATURE_COLS + [
    # Sessions (calendrier)
    "london_active", "newyork_active", "overlap_active", "asia_active",
    "is_monday", "is_friday",
    # Liquidité (proxy volume)
    "volume_ratio_5", "volume_spike",
    # Microstructure (anatomie bougie)
    "body_ratio", "upper_shadow", "lower_shadow", "bull_candle",
    # Régime de volatilité
    "atr_pct",
]


def session_filter(index_or_hour) -> tuple[bool, str]:
    """
    GARDE session : trading autorisé seulement pendant London+NY (7h-21h UTC).

    Recherche 2025 : la session asiatique offre moins de liquidité et des
    signaux plus bruités sur forex/métaux. Boom/Crash tournent 24/24 mais
    le même filtre reste prudent par défaut.

    Retourne (tradeable, raison).
    """
    try:
        if hasattr(index_or_hour, "hour"):
            hour = int(index_or_hour.hour)
        else:
            hour = int(index_or_hour)
    except Exception:
        return True, "heure inconnue → pas de filtre"
    if 7 <= hour < 21:
        return True, "session active (London/NY)"
    return False, f"hors session active ({hour}h UTC — asiatique)"


def _hmm_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features HMM : rendement log + volatilité 20 périodes.
    Retourne un DataFrame propre (NaN éliminés).
    """
    if df is None or df.empty or "close" not in df.columns:
        return pd.DataFrame()
    c = df["close"]
    log_ret = np.log(c / c.shift(1))
    return pd.DataFrame({
        "log_ret": log_ret,
        "volatility": log_ret.rolling(20).std(),
    }).dropna()


# TODO: _vote_from_proba removed — was a trivial pass-through to proba_to_vote() (inlined at LSTMPredictor.predict).

# ── MODÈLE 1 : HMM ──────────────────────────────────────────────────
class HMMPredictor:
    """Baum-Welch training + Forward-Backward + Viterbi prediction."""

    FILE = "hmm_deriv.pkl"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.labels = {}
        self.is_trained = False
        self._load()

    def train(self, df: pd.DataFrame):
        feats = _hmm_features(df)
        if len(feats) < 60:
            raise ValueError(f"HMM : besoin de 60+ bougies propres, reçu {len(feats)}")

        # GARDE-FOU 1 (López de Prado, 2018) : le scaler est fit sur le
        # train (80%) uniquement, puis la série complète est transformée
        # avec ces paramètres. Baum-Welch a besoin de la séquence entière,
        # mais aucune statistique du val n'entre dans la normalisation.
        # Le scaler est sauvegardé dans le pickle avec le modèle.
        split = max(1, int(len(feats) * 0.8))
        self.scaler = RobustScaler()
        self.scaler.fit(feats.values[:split])       # fit sur train SEUL
        X = self.scaler.transform(feats.values)     # transform uniquement

        self.model = GaussianHMM(
            n_components=3,      # Bull / Bear / Range
            covariance_type="full",
            n_iter=500,
            random_state=42,
        )
        self.model.fit(X)

        # Auto-labelling par rendement moyen des états
        states = self.model.predict(X)
        avg_ret = {s: float(feats["log_ret"].values[states == s].mean())
                   for s in range(3)}
        sorted_s = sorted(avg_ret.items(), key=lambda x: x[1])
        self.labels = {
            sorted_s[0][0]: "Bear",
            sorted_s[1][0]: "Range",
            sorted_s[2][0]: "Bull",
        }
        self.is_trained = True
        self._save()
        logger.info(f"HMM entraîné — labels : {self.labels}")

    def predict(self, df: pd.DataFrame) -> dict:
        """Régime actuel + probabilités Forward-Backward + séquence Viterbi."""
        if self.model is None or not self.is_trained:
            return {"regime": "Range", "viterbi": "Range", "confiance": 0.0,
                    "coherent": True, "vote": 0.5, "p_useful": None}

        feats = _hmm_features(df)
        if len(feats) == 0:
            return {"regime": "Range", "viterbi": "Range", "confiance": 0.0,
                    "coherent": True, "vote": 0.5, "p_useful": None}
        if self.scaler is None:  # modèle pré-adapté sans scaler stocké
            self.scaler = RobustScaler().fit(feats.values)

        X = self.scaler.transform(feats.values)

        # Forward-Backward via predict_proba
        probas = self.model.predict_proba(X)
        last = probas[-1]
        best = int(np.argmax(last))

        # Viterbi : séquence d'états la plus probable
        viterbi_states = self.model.predict(X)
        viterbi_last = int(viterbi_states[-1])

        regime = self.labels.get(best, "Range")
        regime_vit = self.labels.get(viterbi_last, "Range")
        confiance = float(last[best])

        # Vote HMM : basé sur la probabilité forward-backward
        bull_state = [s for s, label in self.labels.items() if label == "Bull"]
        bear_state = [s for s, label in self.labels.items() if label == "Bear"]

        proba_bull = sum(last[s] for s in bull_state) if bull_state else 0.0
        proba_bear = sum(last[s] for s in bear_state) if bear_state else 0.0

        if proba_bull > 0.6 and regime_vit == "Bull":
            vote = 1
        elif proba_bear > 0.6 and regime_vit == "Bear":
            vote = 0
        elif proba_bull > 0.5:
            vote = 0.7  # haussier mais pas confirmé à 60%
        elif proba_bear > 0.5:
            vote = 0.3
        else:
            vote = 0.5  # neutre

        return {
            "regime": regime,
            "viterbi": regime_vit,
            "confiance": round(confiance, 3),
            "coherent": regime == regime_vit,
            "vote": vote,
            # P0 — proba_up = P(état haussier) issu du Forward-Backward
            # (somme des probabilités des états labellisés « Bull »).
            # Indispensable au WFO : sans cette clé, `_evaluate` retombait
            # sur le défaut 0.5 → le modèle ne prédisait JAMAIS « down » et
            # son win rate devenait le simple taux de base de l'indice
            # (mesuré : 89.67% sur CRASH500 M1 = zéro compétence).
            "proba_up": round(float(proba_bull), 4),
        }

    def _save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(_model_path(self.FILE), "wb") as f:
            pickle.dump({"model": self.model, "labels": self.labels,
                         "scaler": self.scaler}, f)

    def _load(self):
        path = _model_path(self.FILE)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    d = safe_pickle_load(f)
                    self.model = d.get("model")
                    self.labels = d.get("labels", {})
                    self.scaler = d.get("scaler")
                    self.is_trained = self.model is not None
            except Exception as exc:  # pragma: no cover
                logger.warning(f"HMM : chargement modèle impossible ({exc})")


# ── MODÈLE 2 : XGBoost ──────────────────────────────────────────────
class XGBoostPredictor:
    """
    XGBoost sur 8 features techniques stables (réduit de 15 pour limiter
    le surapprentissage sur les petits datasets H4/H1).

    Améliorations anti-overfitting :
    - `max_depth=3` (au lieu de 4) : arbres moins profonds, plus généraux
    - `min_child_weight=15` : exige plus d'échantillons par feuille
    - Inner `TimeSeriesSplit(3)` : validation temporelle interne
    - `threshold` adaptatif : calibré sur le split de validation
      → remplace le seuil 0.65/0.35 fixe qui générait des folds extrêmes

    Anti-fuite :
    - Utilise `build_labels()` séparé (jamais `feat_df["target"]`)
    - Scaler `fit` uniquement sur X_train, pas sur X_val

    Si xgboost n'est pas installé, le modèle vote neutre (0.5).
    """
    FILE = "xgb_deriv.pkl"
    # Seuils calibrés via calibrate_global_threshold (0.50/0.50 par défaut)

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self.threshold_up   = self._DEFAULT_THR_UP
        self.threshold_down = self._DEFAULT_THR_DOWN
        self._load()

    # ── Calibration du seuil de vote ─────────────────────────────────
    # P0 — SEUIL GLOBAL : plus de per-fold.
    # La calibration se fait sur les probas out-of-fold POOLÉES de tout le
    # WFO (n=150-300) via `calibrate_global_threshold`, pas sur ~40 lignes
    # de validation interne par fold. `train()` initialise donc au seuil
    # neutre ci-dessous (vote = signe de proba - 0.5) ; la règle GLOBALE
    # remplace ces seuils dans `WalkForwardOptimizer` et dans le bot une
    # fois calibrée.
    _DEFAULT_THR_UP   = 0.50
    _DEFAULT_THR_DOWN = 0.50
    MIN_TRADES_CALIBRATION = 30   # en dessous → seuil neutre 0.50/0.50

    @staticmethod
    def calibrate_global_threshold(
        probas: np.ndarray,
        y_true: np.ndarray,
        candidates: list[float] | None = None,
        min_trades: int = 30,
    ) -> tuple[float, float]:
        return calibrate_threshold(probas, y_true, candidates, min_trades)

    def train(self, df: pd.DataFrame):
        if not _XGB_OK:
            logger.warning(
                f"XGBoost indisponible (ignore) : {_XGB_ERROR[:120]}"
            )
            return

        # ── Axe 1 : Features et labels séparés (anti-fuite) ──────────
        # P0-v2 (audit 90 %) : target = triple-barrière symétrique, MÊMES
        # paramètres que le WFO (touch_mult=1.5, horizon=3). La target
        # triviale close[t+1] > close[t] n'apprenait que la dérive de
        # l'indice (acc réelle = acc permutée = baseline UP).
        feat_df = build_features(df)
        y_series = build_labels(
            df, mode="triple_barrier", touch_mult=1.5, horizon=3,
        )
        if feat_df.empty or y_series is None:
            raise ValueError("XGBoost : données insuffisantes pour build_features/labels")

        # Alignement chronologique strict sur l'index commun
        common_idx = feat_df.index.intersection(y_series.index)
        if len(common_idx) < 80:
            raise ValueError(
                f"XGBoost : besoin de 80+ lignes alignées, reçu {len(common_idx)}"
            )
        feat_df = feat_df.loc[common_idx]
        y_series = y_series.loc[common_idx]

        X_raw = feat_df[FEATURE_COLS].values
        y     = y_series.values

        # ── GARDE-FOU 1 (López de Prado, 2018) : split AVANT le fit ───
        # Le scaler est fit UNIQUEMENT sur le train (80%) ; la validation
        # est transformée avec les paramètres du train. Aucune statistique
        # (médiane/IQR) du futur ne remonte dans le passé.
        # P0 : le split interne 80/20 sert UNIQUEMENT au garde-fou scaler
        # (médiane/IQR du train). Le FIT XGB se fait sur 100% du train du
        # fold — pas de split d'entraînement + pas de calibration per-fold.
        split = int(len(X_raw) * 0.8)
        X_tr_raw = X_raw[:split]

        self.scaler = RobustScaler()
        self.scaler.fit(X_tr_raw)                    # fit sur train SEUL
        X_scaled_all = self.scaler.transform(X_raw)  # transform uniquement
        # P0 : le check "target monolithique" porte sur y (100% du fold),
        # pas sur un split partiel supprimé.
        if len(np.unique(y)) < 2:
            raise ValueError("XGBoost : target monolithique, impossible d'entraîner")

        # ── Axe 2 : Hyperparamètres plus conservateurs (anti-overfitting)
        kwargs = dict(
            n_estimators=300,
            max_depth=3,           # 4→3 : arbres moins profonds
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            reg_alpha=1.0,
            reg_lambda=2.0,
            min_child_weight=15,   # 10→15 : plus d'échantillons par feuille
            gamma=0.2,             # 0.1→0.2 : coût de division plus élevé
        )
        if int(_xgb.__version__.split(".")[0]) < 2:
            kwargs["use_label_encoder"] = False  # type: ignore

        self.model = XGBClassifier(**kwargs)
        # P0 — PAS de fit partiel + calibration per-fold sur ~40 lignes :
        # fit sur TOUT le train du fold, seuils NEUTRES (0.50/0.50).
        # La vraie règle est le SEUIL GLOBAL calibré sur les probas
        # out-of-fold POOLÉES de tout le WFO (n=150-300) via
        # `calibrate_global_threshold`. L'accuracy ci-dessous est
        # diagnostique (train) et ne sert pas à calibrer.
        self.threshold_up, self.threshold_down = 0.50, 0.50
        try:
            self.model.fit(X_scaled_all, y)
        except TypeError:
            self.model.fit(X_scaled_all, y)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, self.model.predict(X_scaled_all))

        self.is_trained = True
        self._save()
        logger.info(
            f"XGBoost entraîné — accuracy train : {acc:.1%} | "
            f"seuils neutres ↑0.50 ↓0.50 (calibration globale en WFO)"
        )

    def predict_proba(self, df: pd.DataFrame) -> float | None:
        """
        Probabilité brute P(up) SANS application de seuil.

        Séparation calibration / décision (P0) : le WFO collecte ces probas
        brutes (out-of-fold poolées + in-sample) puis calibre le SEUIL GLOBAL
        via `calibrate_global_threshold`. `predict()` reste le wrapper seuillé
        utilisé en production. Retourne None si indisponible / insuffisant.
        """
        if self.model is None or not self.is_trained or not _XGB_OK:
            return None

        feat_df = build_features(df)
        if feat_df.empty or not set(FEATURE_COLS).issubset(feat_df.columns):
            return None

        X = feat_df[FEATURE_COLS].dropna()
        if X.empty or self.scaler is None:
            return None

        # GARDE-FOU 1 : transform() uniquement — jamais de refit en production.
        X_scaled = self.scaler.transform(X.iloc[[-1]].values)
        return float(self.model.predict_proba(X_scaled)[0][1])

    def predict(self, df: pd.DataFrame) -> dict:
        proba = self.predict_proba(df)
        if proba is None:
            return {"proba_up": 0.5, "vote": 0.5}
        proba = float(proba)
        # Vote : seuils de l'instance (neutres 0.50/0.50 en WFO — la
        # calibration GLOBALE poolée remplace ces seuils dans le WFO).
        if proba >= self.threshold_up:
            vote = 1
        elif proba <= self.threshold_down:
            vote = 0
        else:
            vote = 0.5
        return {"proba_up": round(proba, 3), "vote": vote}

    def _save(self):
        with open(_model_path(self.FILE), "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "threshold_up": self.threshold_up,
                "threshold_down": self.threshold_down,
            }, f)

    def _load(self):
        path = _model_path(self.FILE)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    d = safe_pickle_load(f)
                    self.model = d.get("model")
                    self.scaler = d.get("scaler")
                    self.threshold_up   = d.get("threshold_up",   self._DEFAULT_THR_UP)
                    self.threshold_down = d.get("threshold_down", self._DEFAULT_THR_DOWN)
                    self.is_trained = self.model is not None and _XGB_OK
            except Exception as exc:  # pragma: no cover
                logger.warning(f"XGBoost : chargement modèle impossible ({exc})")


# ── MODÈLE 3 : LSTM ─────────────────────────────────────────────────
class LSTMPredictor:
    """
    LSTM 3 couches pour capturer les dépendances temporelles longues.
    Architecture prouvée sur crypto/forex : MAPE 17% vs GARCH 125%.
    Window = 30 bougies → prédit la direction de la prochaine.

    Requiert tensorflow. S'il est absent, le modèle vote neutre (0.5).
    """
    FILE = "lstm_deriv.h5"
    SCALER_FILE = "lstm_scaler.pkl"
    WINDOW = 30

    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        self._load()

    def train(self, df: pd.DataFrame):
        if not _TF_OK:
            logger.warning(
                f"LSTM indisponible (ignore) : tensorflow manquant ({_TF_ERROR[:80]}). "
                "pip install tensorflow pour activer le 3e modèle."
            )
            return
        feat_df = build_features(df)
        # P0-v2 : triple-barrière symétrique (cohérence avec le WFO et
        # XGBoost — voir XGBoostPredictor.train).
        y_series = build_labels(
            df, mode="triple_barrier", touch_mult=1.5, horizon=3,
        )
        if y_series is None or len(feat_df) < self.WINDOW + 40:
            raise ValueError(
                f"LSTM : besoin de {self.WINDOW + 40}+ lignes features, reçu {len(feat_df)}"
            )

        # Alignement strict features / labels (anti-fuite)
        common_idx = feat_df.index.intersection(y_series.index)
        feat_df    = feat_df.loc[common_idx]
        y_series   = y_series.loc[common_idx]

        X_raw = feat_df[FEATURE_COLS].values
        y_raw = y_series.values

        # GARDE-FOU 1 (López de Prado, 2018) : scaler fit sur le train
        # uniquement (80%). Les séquences à cheval sur la frontière utilisent
        # des features transformées avec les statistiques du train → aucune
        # fuite de la validation vers l'entraînement.
        split_raw = int(len(X_raw) * 0.8)
        self.scaler = RobustScaler()
        self.scaler.fit(X_raw[:split_raw])          # fit sur train SEUL
        X_scaled = self.scaler.transform(X_raw)     # transform uniquement

        # Séquences de WINDOW bougies
        X_seq, y_seq = [], []
        for i in range(self.WINDOW, len(X_scaled)):
            X_seq.append(X_scaled[i - self.WINDOW:i])
            y_seq.append(y_raw[i])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        split = int(len(X_seq) * 0.8)
        X_tr, X_val = X_seq[:split], X_seq[split:]
        y_tr, y_val = y_seq[:split], y_seq[split:]

        # Modèle LSTM bi-directionnel
        inp = tf.keras.Input(shape=(self.WINDOW, len(FEATURE_COLS)))
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True))(inp)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.LSTM(32, return_sequences=False)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        out = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        self.model = tf.keras.Model(inp, out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        cb = tf.keras.callbacks.EarlyStopping(
            patience=10, restore_best_weights=True
        )
        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[cb],
            verbose=0,
        )

        acc = float(self.model.evaluate(X_val, y_val, verbose=0)[1])
        self.is_trained = True
        self.model.save(_model_path(self.FILE))
        with open(_model_path(self.SCALER_FILE), "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"LSTM entraîné — accuracy validation : {acc:.1%}")

    def predict(self, df: pd.DataFrame) -> dict:
        if self.model is None or not self.is_trained or not _TF_OK:
            return {"proba_up": 0.5, "vote": 0.5}

        feat_df = build_features(df)
        X = feat_df[FEATURE_COLS].dropna()
        if len(X) < self.WINDOW or self.scaler is None:
            return {"proba_up": 0.5, "vote": 0.5}

        X_scaled = self.scaler.transform(X.values)
        X_seq = X_scaled[-self.WINDOW:].reshape(1, self.WINDOW, -1)
        proba = float(self.model.predict(X_seq, verbose=0)[0][0])

        # TODO: _vote_from_proba was a trivial pass-through to proba_to_vote() - inlined (removed function).
        vote = proba_to_vote(proba)
        return {"proba_up": round(proba, 3), "vote": vote}

    def _load(self):
        if not _TF_OK:
            return
        path = _model_path(self.FILE)
        scalar_path = _model_path(self.SCALER_FILE)
        if os.path.exists(path):
            try:
                self.model = tf.keras.models.load_model(path)
                self.is_trained = self.model is not None
            except Exception as exc:  # pragma: no cover
                logger.warning(f"LSTM : chargement modèle impossible ({exc})")
                self.model = None
                self.is_trained = False
        if os.path.exists(scalar_path):
            try:
                with open(scalar_path, "rb") as f:
                    self.scaler = safe_pickle_load(f)
            except Exception:  # pragma: no cover
                self.scaler = None


# ── MODÈLE 4 : KALMAN MOMENTUM ──────────────────────────────────────
class KalmanMomentumPredictor:
    """
    Vote basé sur le momentum du prix filtré par Kalman.
    Simple mais robuste : est-ce que le prix filtré monte ou descend ?
    """
    def predict(self, df: pd.DataFrame) -> dict:
        if len(df) < 6:
            return {"trend": 0.0, "mom": 0.0, "vote": 0.5}

        close = df["close"]
        filtered = _kalman_filter(close)

        # Tendance sur les 5 dernières bougies filtrées
        trend = filtered.iloc[-1] - filtered.iloc[-5]
        # Momentum sur 3 bougies
        mom = filtered.iloc[-1] - filtered.iloc[-3]

        if trend > 0 and mom > 0:
            vote = 1   # BUY
        elif trend < 0 and mom < 0:
            vote = 0   # SELL
        else:
            vote = 0.5  # HOLD

        return {
            "trend": round(float(trend), 4),
            "mom": round(float(mom), 4),
            "vote": vote,
        }


# ── MODÈLE 5 : RSI DIVERGENCE ───────────────────────────────────────
class RSIDivergencePredictor:
    """
    Détecte les divergences RSI/Prix — signal de retournement fort.
    Divergence classique :
      haussière  = plus bas prix plus bas + RSI plus haut,
      baissière  = plus haut prix plus haut + RSI plus bas.
    Simple et prouvé en trading manuel et algorithmique.
    """
    WINDOW = 10

    def predict(self, df: pd.DataFrame) -> dict:
        if len(df) < self.WINDOW + 2:
            return {"rsi": 50.0, "divergence_bull": False,
                    "divergence_bear": False, "vote": 0.5}

        close = df["close"]
        rsi = ta.momentum.RSIIndicator(close, 14).rsi()

        window = min(self.WINDOW, len(close) - 1)
        recent_prices = close.iloc[-window:]
        recent_rsi = rsi.iloc[-window:]

        if len(recent_prices) < 3 or recent_rsi.notna().sum() < 3:
            return {"rsi": float(rsi.iloc[-1]), "divergence_bull": False,
                    "divergence_bear": False, "vote": 0.5}

        # Deux zones temporelles : lointain (cheville) vs récent (pivot)
        pivot_len = max(3, window // 2)
        pivot_prices = recent_prices.iloc[:pivot_len]
        tail_prices = recent_prices.iloc[pivot_len:]
        if pivot_prices.empty or tail_prices.empty:
            return {"rsi": float(rsi.iloc[-1]), "divergence_bull": False,
                    "divergence_bear": False, "vote": 0.5}

        # --- Divergence haussière (bullish) ---
        low_pivot_idx = pivot_prices.idxmin()
        low_tail_idx = tail_prices.idxmin()
        rsi_pivot = recent_rsi.loc[low_pivot_idx]
        rsi_tail = recent_rsi.loc[low_tail_idx]
        divergence_bull = bool(
            tail_prices.loc[low_tail_idx] < pivot_prices.loc[low_pivot_idx]
            and pd.notna(rsi_pivot) and pd.notna(rsi_tail)
            and rsi_tail > rsi_pivot
        )

        # --- Divergence baissière (bearish) ---
        high_pivot_idx = pivot_prices.idxmax()
        high_tail_idx = tail_prices.idxmax()
        rsi_pivot_h = recent_rsi.loc[high_pivot_idx]
        rsi_tail_h = recent_rsi.loc[high_tail_idx]
        divergence_bear = bool(
            tail_prices.loc[high_tail_idx] > pivot_prices.loc[high_pivot_idx]
            and pd.notna(rsi_pivot_h) and pd.notna(rsi_tail_h)
            and rsi_tail_h < rsi_pivot_h
        )

        rsi_val = float(rsi.iloc[-1])

        # Detecter la tendance ADX pour eviter les faux signaux
        adx_val = 0.0
        try:
            if len(close) >= 20:
                adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], close, 14)
                adx_val = float(adx_ind.adx().iloc[-1])
        except Exception:
            adx_val = 0.0

        # Tendance haussiere forte (ADX > 25 + prix > EMA50)
        ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
        trend_up = adx_val > 25 and close.iloc[-1] > ema50.iloc[-1]
        trend_down = adx_val > 25 and close.iloc[-1] < ema50.iloc[-1]

        # Logique adaptée au régime
        if trend_up:
            # Tendance haussière forte : RSI > 75 = continuation, RSI < 30 = pullback acheteur
            if divergence_bull:
                vote = 1
            elif rsi_val > 70:
                vote = 0.8  # continuation haussière
            elif rsi_val < 30:
                vote = 0.7  # pullback en tendance haussière = achat
            else:
                vote = 0.6  # légèrement haussier par défaut
        elif trend_down:
            # Tendance baissière forte : RSI < 25 = continuation, RSI > 70 = rallye vendeur
            if divergence_bear:
                vote = 0
            elif rsi_val < 30:
                vote = 0.2  # continuation baissière
            elif rsi_val > 70:
                vote = 0.3  # rallye en tendance baissière = vente
            else:
                vote = 0.4  # légèrement baissier par défaut
        else:
            # Range : logique divergence classique
            if divergence_bull and not trend_down:
                vote = 1
            elif divergence_bear and not trend_up:
                vote = 0
            elif rsi_val < 25 and not trend_down:
                vote = 0.6
            elif rsi_val > 75 and not trend_up:
                vote = 0.4
            else:
                vote = 0.5

        return {
            "rsi": round(rsi_val, 1),
            "adx": round(adx_val, 1),
            "divergence_bull": divergence_bull,
            "divergence_bear": divergence_bear,
            "vote": vote,
        }


# ── MODÈLE 6 : TREND STRENGTH (EMA ALIGNMENT + ADX) ──────────────────

# ── MODÈLE 7 : MOMENTUM TREND ────────────────────────────────────────
class MomentumPredictor:
    """
    Momentum simple et robuste : ROC + alignement EMA.
    - ROC(10) > seuil et EMA9 > EMA21 → vote 1 (BUY)
    - ROC(10) < -seuil et EMA9 < EMA21 → vote 0 (SELL)
    - sinon → vote 0.5 (neutre)
    
    Le seuil est basé sur l'ATR relatif pour éviter les faux signaux sur bruit.
    """
    def predict(self, df: pd.DataFrame) -> dict:
        if len(df) < 25:
            return {"roc": 0.0, "vote": 0.5}

        close = df["close"]
        roc = ta.momentum.ROCIndicator(close, 10).roc()
        ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()

        r = float(roc.iloc[-1])
        e9 = float(ema9.iloc[-1])
        e21 = float(ema21.iloc[-1])

        # Seuil de significativité basé sur l'ATR relatif
        atr_pct = 0.0
        try:
            atr = ta.volatility.AverageTrueRange(df["high"], df["low"], close, 14).average_true_range()
            atr_pct = float(atr.iloc[-1]) / float(close.iloc[-1])
        except Exception:
            atr_pct = 0.001  # fallback 0.1%

        # Seuil strict : momentum doit dépasser l'ATR pour être significatif
        seuil = max(atr_pct * 1.0, 0.003)

        if r > seuil and e9 > e21:
            vote = 1
        elif r < -seuil and e9 < e21:
            vote = 0
        else:
            vote = 0.5

        return {"roc": round(r, 4), "vote": vote}


class TrendStrengthPredictor:
    """
    Force de la tendance via alignement EMA + ADX.
    - EMA9 > EMA21 > EMA50 → tendance haussière structurelle
    - EMA9 < EMA21 < EMA50 → tendance baissière structurelle
    - ADX > 18 → tendance confirmée (seuil assouplifié)
    
    Filtre anti-bruit : écart EMA9-EMA50 significatif.
    Vote binaire : 1 (BUY) ou 0 (SELL) quand tendance claire, 0.5 sinon.
    """
    def predict(self, df: pd.DataFrame) -> dict:
        if len(df) < 55:
            return {"adx": 0.0, "ema_align": "NONE", "vote": 0.5}

        close = df["close"]
        ema9  = ta.trend.EMAIndicator(close, 9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
        ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()

        adx_val = 0.0
        try:
            adx_ind = ta.trend.ADXIndicator(df["high"], df["low"], close, 14)
            adx_val = float(adx_ind.adx().iloc[-1])
        except Exception:
            adx_val = 0.0

        e9, e21, e50 = ema9.iloc[-1], ema21.iloc[-1], ema50.iloc[-1]
        prix = float(close.iloc[-1])

        # Filtre anti-bruit : écart EMA9-EMA50 significatif (seuil assoupli)
        atr = 0.0
        try:
            atr = float(ta.volatility.AverageTrueRange(df["high"], df["low"], close, 14).average_true_range().iloc[-1])
        except Exception:
            atr = prix * 0.001

        ema_ecart = abs(e9 - e50)
        ema_significatif = ema_ecart > (atr * 0.2)

        # Alignement EMA : condition structurelle principale
        bull_align = e9 > e21 > e50
        bear_align = e9 < e21 < e50

        # Filtre de pente EMA50 significative (anti-bruit)
        ema50_vals = ema50.values if hasattr(ema50, "values") else np.array(ema50)
        if len(ema50_vals) >= 20:
            pente_ema50 = float(ema50_vals[-1] - ema50_vals[-20]) / 20.0
        else:
            pente_ema50 = float(ema50_vals[-1] - ema50_vals[0]) / max(len(ema50_vals) - 1, 1)
        pente_significative = abs(pente_ema50) > (atr * 0.15)

        if bull_align and ema_significatif and pente_significative:
            align = "BULLISH"
            base_vote = 1
        elif bear_align and ema_significatif and pente_significative:
            align = "BEARISH"
            base_vote = 0
        else:
            align = "MIXED"
            base_vote = 0.5

        # Vote binaire : 1 ou 0 si tendance claire, 0.5 sinon
        if align != "MIXED":
            vote = float(base_vote)
            confidence = min(adx_val / 45, 1.0) if adx_val > 18 else 0.5
        else:
            vote = 0.5
            confidence = 0.0

        return {
            "adx": round(adx_val, 1),
            "ema_align": align,
            "confidence": round(confidence, 2),
            "vote": vote,
        }


# ── ENSEMBLE : VOTE MAJORITAIRE QUALIFIÉ ────────────────────────────
class EnsemblePredictor:
    """
    Croise les 5+ modèles.
    Décision finale = vote à la majorité QUALIFIÉE (4/5 minimum).

    Modèles actifs par défaut : HMM, XGB, Kalman, RSI, TrendStrength, Momentum
    LSTM s'ajoute automatiquement si TensorFlow est disponible.

    RÈGLE ABSOLUE : un trade ne se place QUE si au moins 4 modèles sur 5
    sont d'accord. En dessous de 4/5 → HOLD. Toujours.
    """

    MIN_VOTES_TO_TRADE = 4  # NE PAS TOUCHER — règle absolue du projet

    def __init__(self):
        self.hmm = HMMPredictor()
        self.xgb = XGBoostPredictor()
        self.lstm = LSTMPredictor()
        self.kalman = KalmanMomentumPredictor()
        self.rsi = RSIDivergencePredictor()
        self.trend = TrendStrengthPredictor()
        self.momentum = MomentumPredictor()

    def is_trained(self) -> bool:
        """Au moins un modèle supervisé (HMM/XGB/LSTM) est disponible."""
        return any([
            getattr(self.hmm, "is_trained", False),
            getattr(self.xgb, "is_trained", False),
            getattr(self.lstm, "is_trained", False),
        ])

    def predict(self, df: pd.DataFrame) -> dict:
        """
        Lance les 7 modèles et retourne la décision finale.

        Modèles actifs : HMM, XGB, LSTM (si TF), Kalman, RSI, TrendStrength, Momentum
        Décision = majorité qualifiée 4/5 (MIN_VOTES_TO_TRADE).

        Retourne :
          signal      : BUY | SELL | HOLD
          confiance   : fraction de modèles d'accord (0.0 si HOLD)
          buy_votes / sell_votes / hold_votes
          detail      : détails par modèle
          tradeable   : True seulement si BUY/SELL → exécution possible
        """
        res_hmm    = self.hmm.predict(df)
        res_xgb    = self.xgb.predict(df)
        res_lstm   = self.lstm.predict(df)
        res_kalman = self.kalman.predict(df)
        res_rsi    = self.rsi.predict(df)
        res_trend  = self.trend.predict(df)
        res_momentum = self.momentum.predict(df)

        votes = {
            "HMM":           res_hmm["vote"],
            "XGBoost":       res_xgb["vote"],
            "LSTM":          res_lstm["vote"],
            "Kalman":        res_kalman["vote"],
            "RSI":           res_rsi["vote"],
            "TrendStrength": res_trend["vote"],
            "Momentum":      res_momentum["vote"],
        }

        # Compte les votes BUY (1), SELL (0), neutres (0.5 ou intermédiaires)
        buy_votes  = sum(1 for v in votes.values() if v >= 0.7)
        sell_votes = sum(1 for v in votes.values() if v <= 0.3)
        hold_votes = sum(1 for v in votes.values() if 0.3 < v < 0.7)

        n_models = len(votes)

        # Décision à la majorité qualifiée (4/5 minimum → 4 modèles d'accord)
        if buy_votes >= self.MIN_VOTES_TO_TRADE:
            signal     = "BUY"
            confiance  = buy_votes / n_models
        elif sell_votes >= self.MIN_VOTES_TO_TRADE:
            signal     = "SELL"
            confiance  = sell_votes / n_models
        else:
            signal     = "HOLD"
            confiance  = 0.0

        return {
            "signal":     signal,
            "confiance":  round(confiance, 2),
            "buy_votes":  buy_votes,
            "sell_votes": sell_votes,
            "hold_votes": hold_votes,
            "detail": {
                "HMM":           {"regime":    res_hmm.get("regime", "N/A"),
                                  "confiance": res_hmm.get("confiance", 0.0),
                                  "vote":      votes["HMM"]},
                "XGBoost":       {"proba":     res_xgb["proba_up"],         "vote": votes["XGBoost"]},
                "LSTM":          {"proba":     res_lstm["proba_up"],        "vote": votes["LSTM"]},
                "Kalman":        {"trend":     res_kalman["trend"],         "vote": votes["Kalman"]},
                "RSI":           {"rsi":       res_rsi["rsi"],              "vote": votes["RSI"]},
                "TrendStrength": {"ema_align": res_trend["ema_align"],      "vote": votes["TrendStrength"]},
                "Momentum":      {"roc":       res_momentum["roc"],         "vote": votes["Momentum"]},
            },
            "tradeable": signal in ("BUY", "SELL"),
        }

    def train_all(self, df: pd.DataFrame):
        """Entraîne tous les modèles sur le même dataset (chronologique)."""
        logger.info("=== Entraînement HMM ===")
        self.hmm.train(df)
        logger.info("=== Entraînement XGBoost ===")
        self.xgb.train(df)
        logger.info("=== Entraînement LSTM ===")
        self.lstm.train(df)
        logger.info("=== Entraînement terminé ===")


