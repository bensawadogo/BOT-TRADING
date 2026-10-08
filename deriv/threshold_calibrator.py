"""Calibration de seuils de vote (centrale)."""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def calibrate_threshold(
    probas,
    y_true,
    candidates=None,
    min_trades: int = 30,
    neutral_fallback: tuple[float, float] = (0.60, 0.40),
) -> tuple[float, float]:
    """Calibre un seuil symetrique (thr_up, thr_down) sur des predictions poolees.

    Cherche le seuil qui maximise la balanced accuracy (WR_up + WR_down)/2.
    En dessous de min_trades trades filtres -> seuil neutre (0.50/0.50).
    Ne pas aller au-dessus de 0.75 (causerait 0 trade).
    Retourne : (thr_up, thr_down)
    """
    probas = np.asarray(probas, dtype=float)
    y_true = np.asarray(y_true, dtype=int)

    # GARDE-FOU : desalignement probas/actuals ne doit JAMAIS
    # faire planter tout le systeme. Tronquer sur la longueur commune.
    if len(probas) != len(y_true):
        n_comm = min(len(probas), len(y_true))
        logger.warning(
            "Pool desaligne : %d probas vs %d actuals -> troncature a %d",
            len(probas), len(y_true), n_comm,
        )
        probas, y_true = probas[:n_comm], y_true[:n_comm]

    if candidates is None:
        candidates = [round(0.55 + i * 0.025, 3) for i in range(11)]

    best_score, best_thr = -1.0, 0.50
    for thr in candidates:
        mask = (probas >= thr) | (probas <= (1 - thr))
        if mask.sum() < min_trades:
            continue
        preds = (probas[mask] >= thr).astype(int)
        yt = y_true[mask]
        up_m = preds == 1
        dn_m = preds == 0
        n_up, n_dn = int(up_m.sum()), int(dn_m.sum())
        if n_up < 5 or n_dn < 5:
            continue
        wr_up = float((yt[up_m] == 1).mean())
        wr_down = float((yt[dn_m] == 0).mean())
        score = (wr_up + wr_down) / 2.0
        if score > best_score:
            best_score, best_thr = score, thr

    best_thr = min(best_thr, 0.75)
    if best_score <= 0.0:
        return neutral_fallback
    return best_thr, round(1 - best_thr, 3)