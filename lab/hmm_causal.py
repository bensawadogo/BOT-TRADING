"""
Couche RÉGIME : HMM gaussien avec les trois algorithmes, utilisés sans regard vers le futur.

- Baum-Welch (EM, `hmmlearn.GaussianHMM.fit`) : estime les paramètres, sur une
  fenêtre PASSÉE seulement, réestimée à intervalle fixe (walk-forward).
- Forward (filtre de Hamilton) : P(état_t | x_1..x_t). C'est la seule probabilité
  causale. `predict_proba` de hmmlearn renvoie les probabilités LISSÉES
  (forward-backward), qui utilisent x_{t+1..T} : à ne jamais utiliser comme signal
  historique.
- Viterbi : chemin le plus probable. On ne garde que le DERNIER état d'un décodage
  fait sur les données <= t (fenêtre glissante) ; le chemin complet sur toute la
  série « réécrit » le passé avec le futur.

Variante : modèle à sauts (Statistical Jump Model, Nystrup/Bemporad, paquet
`jumpmodels`, Apache-2.0), via son `predict_online` causal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def features(close: pd.Series, vol_win: int = 20, mom_win: int = 10) -> pd.DataFrame:
    """Rendement log, volatilité et momentum : chacun ne dépend que des clôtures <= t."""
    r = np.log(close).diff()
    return pd.DataFrame({
        "ret": r,
        "vol": r.rolling(vol_win).std(),
        "mom": r.rolling(mom_win).sum(),
    }, index=close.index)


def fit_baum_welch(X: np.ndarray, n_states: int = 3, n_init: int = 3,
                   seed: int = 0) -> GaussianHMM | None:
    """Baum-Welch avec plusieurs initialisations ; garde la meilleure vraisemblance."""
    best, best_ll = None, -np.inf
    for k in range(n_init):
        m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200,
                        tol=1e-4, random_state=seed + k)
        try:
            m.fit(X)
            ll = m.score(X)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    return best


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Filtre forward (version normalisée de Rabiner) : P(état_t | x_1..x_t) pour chaque t.
    Égal à la dernière ligne de `predict_proba(X[:t+1])` (vérifié dans les tests)."""
    log_b = model._compute_log_likelihood(X)       # log p(x_t | état)
    B = np.exp(log_b - log_b.max(axis=1, keepdims=True))
    A = model.transmat_
    out = np.empty_like(B)
    alpha = model.startprob_ * B[0]
    out[0] = alpha / alpha.sum()
    for t in range(1, len(X)):
        alpha = (out[t - 1] @ A) * B[t]
        out[t] = alpha / alpha.sum()
    return out


def viterbi_endpoints(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """Récursion max-produit de Viterbi : à chaque t, argmax de delta_t, c'est-à-dire
    le dernier état du chemin Viterbi calculé sur x_1..x_t (causal, une seule passe)."""
    log_b = model._compute_log_likelihood(X)
    log_A = np.log(np.maximum(model.transmat_, 1e-300))
    d = np.log(np.maximum(model.startprob_, 1e-300)) + log_b[0]
    out = np.empty(len(X), dtype=int)
    out[0] = int(d.argmax())
    for t in range(1, len(X)):
        d = (d[:, None] + log_A).max(axis=0) + log_b[t]
        d -= d.max()
        out[t] = int(d.argmax())
    return out


def _labels(model: GaussianHMM, ret_col: int = 0) -> np.ndarray:
    """Étiquette chaque état par son rendement moyen : -1 baissier, 0 neutre, +1 haussier."""
    mu = model.means_[:, ret_col]
    order = np.argsort(mu)
    lab = np.zeros(len(mu), dtype=int)
    lab[order[0]], lab[order[-1]] = -1, 1
    return lab


def walk_forward_hmm(close: pd.Series, train_len: int = 1000, refit_every: int = 63,
                     n_states: int = 3,
                     cols: tuple[str, ...] = ("ret", "vol", "mom")) -> pd.DataFrame:
    """Régime causal bougie par bougie.

    Colonnes : p_bull, p_bear (filtre forward), state (étiquette filtrée -1/0/+1),
    viterbi (dernier état du chemin Viterbi calculé sur la fenêtre passée).
    Tout ce qui est écrit à la ligne t n'utilise que close[:t+1].
    """
    F = features(close)[list(cols)]
    n = len(F)
    out = pd.DataFrame(np.nan, index=close.index, columns=["p_bull", "p_bear", "state", "viterbi"])
    valid = F.notna().all(axis=1).to_numpy()
    first = int(np.argmax(valid)) if valid.any() else n
    t0 = first + train_len
    if t0 >= n:
        return out
    Xall = F.to_numpy(float)
    for a in range(t0, n, refit_every):
        b = min(a + refit_every, n)
        tr = Xall[a - train_len:a]
        mu, sd = tr.mean(0), tr.std(0) + 1e-12            # normalisation : passé seul
        model = fit_baum_welch((tr - mu) / sd, n_states)
        if model is None:
            continue
        lab = _labels(model)
        Z = (Xall[a - train_len:b] - mu) / sd              # passé + segment à venir
        post = forward_filter(model, Z)                 # récursion causale
        post = post[train_len:]                            # lignes a..b-1
        out.iloc[a:b, 0] = post[:, lab == 1].sum(1)
        out.iloc[a:b, 1] = post[:, lab == -1].sum(1)
        out.iloc[a:b, 2] = lab[post.argmax(1)]
        out.iloc[a:b, 3] = lab[viterbi_endpoints(model, Z)[train_len:]]
    return out


def walk_forward_jump(close: pd.Series, train_len: int = 1000, refit_every: int = 63,
                      n_states: int = 3, jump_penalty: float = 50.0,
                      cols: tuple[str, ...] = ("ret", "vol", "mom")) -> pd.DataFrame:
    """Même interface avec un modèle à sauts (jumpmodels.JumpModel, predict_online)."""
    from jumpmodels.jump import JumpModel

    F = features(close)[list(cols)]
    n = len(F)
    out = pd.DataFrame(np.nan, index=close.index, columns=["state"])
    valid = F.notna().all(axis=1).to_numpy()
    first = int(np.argmax(valid)) if valid.any() else n
    t0 = first + train_len
    Xall = F.to_numpy(float)
    for a in range(t0, n, refit_every):
        b = min(a + refit_every, n)
        tr = Xall[a - train_len:a]
        mu, sd = tr.mean(0), tr.std(0) + 1e-12
        jm = JumpModel(n_components=n_states, jump_penalty=jump_penalty, random_state=0, n_init=3)
        jm.fit((tr - mu) / sd, ret_ser=tr[:, 0], sort_by="cumret")
        # sort_by="cumret" : état 0 = rendement cumulé le plus élevé
        Z = (Xall[a - train_len:b] - mu) / sd
        st = np.asarray(jm.predict_online(Z))[train_len:]
        lab = np.where(st == 0, 1, np.where(st == n_states - 1, -1, 0))
        out.iloc[a:b, 0] = lab
    return out
