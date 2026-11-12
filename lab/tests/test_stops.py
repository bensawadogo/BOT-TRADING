"""La couche de stops du labo doit déclencher quand il faut et ne jamais voir le futur."""
import numpy as np
import pandas as pd

from lab.stops import DELAI, Superposition, Variante


def _base(signals, vols, returns):
    return pd.Series({"A": 0.5})


VOL = pd.Series({"A": 0.10})        # 10 %/an : un σ mensuel vaut ≈ 2,9 %
SIG = pd.Series({"A": 1.0})


def _rend(x):
    return pd.DataFrame({"A": np.asarray(x, float)})


def _joue(variante, rendements):
    sup = Superposition(_base, variante)
    sorties = []
    for i in range(1, len(rendements) + 1):
        sorties.append(float(sup(SIG, VOL, _rend(rendements[:i]))["A"]))
    return sup, sorties


# +1 % pendant 5 jours (gain 5 %), puis −1 % pendant 8 jours
CHEMIN = [0.01] * 5 + [-0.01] * 8


def test_suiveur_coupe_apres_le_recul_depuis_le_plus_haut():
    v = Variante("t", "suiveur", 1.5, "signal")          # distance ≈ 4,3 %
    sup, sortie = _joue(v, CHEMIN)
    assert sortie[:9] == [0.5] * 9                       # recul de 4 % au plus : on tient
    assert sortie[9] == 0.0 and sup.declenchements == 1  # recul de 5 % : stop


def test_stop_depuis_l_entree_ne_coupe_pas_une_position_encore_gagnante():
    v = Variante("t", "entree", 1.5, "signal")
    sup, sortie = _joue(v, CHEMIN)                       # le cumul finit à −4 %, sous la distance de 4,3 %
    assert sup.declenchements == 0 and set(sortie) == {0.5}


def test_retour_apres_delai_et_attente_d_un_nouveau_signal():
    plat = [0.0] * 40
    delai = Variante("t", "suiveur", 1.5, "delai")
    _, s_delai = _joue(delai, CHEMIN + plat)
    assert s_delai[9] == 0.0 and s_delai[-1] == 0.5      # revient une fois DELAI jours écoulés
    assert s_delai[9 + DELAI - 2] == 0.0
    signal = Variante("t", "suiveur", 1.5, "signal")
    _, s_signal = _joue(signal, CHEMIN + plat)
    assert s_signal[-1] == 0.0                           # le signal n'a pas changé : reste à plat


def test_le_futur_ne_change_pas_le_passe():
    v = Variante("t", "suiveur", 1.5, "delai")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 200)
    _, long = _joue(v, x)
    _, court = _joue(v, x[:120])
    assert long[:120] == court
