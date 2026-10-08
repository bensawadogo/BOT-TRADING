"""Fonctions centralisees de vote (proba <-> vote)."""
from __future__ import annotations


def proba_to_vote(proba: float, threshold_up: float = 0.65, threshold_down: float = 0.35) -> int | float:
    """Convertit une proba directionnelle en vote discret.

    Vote : 1 si proba > threshold_up, 0 si proba < threshold_down, sinon 0.5 (neutre).
    """
    if proba > threshold_up:
        return 1
    if proba < threshold_down:
        return 0
    return 0.5


def vote_to_proba(vote) -> float | None:
    """Convertit un vote discret en proba directionnelle.

    Un vote NEUTRE (0.5) ne porte AUCUNE information : il est rejete (None)
    au lieu d'etre converti en 0.5, sinon le seuillage (proba >= thr) le
    ferait passer pour un "UP" permanent et le win rate mesurerait la
    derive de l'indice au lieu de la competence.
    """
    if isinstance(vote, bool) or not isinstance(vote, (int, float)):
        return None
    v = float(vote)
    if v >= 0.7 or v <= 0.3:
        return v
    return None