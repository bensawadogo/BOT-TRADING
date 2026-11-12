"""Le dashboard s'affiche sans erreur, avec ou sans journal, et n'invente rien."""
import json
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def _lancer(monkeypatch, tmp_path):
    monkeypatch.setenv("TREND_JOURNAL", str(tmp_path / "trend_journal.csv"))
    monkeypatch.setenv("TREND_STATE", str(tmp_path / "trend_state.json"))
    return AppTest.from_file(APP, default_timeout=30).run()


def test_sans_journal(monkeypatch, tmp_path):
    at = _lancer(monkeypatch, tmp_path)
    assert not at.exception
    assert "Aucun cycle" in at.info[0].value


def test_avec_journal(monkeypatch, tmp_path):
    (tmp_path / "trend_journal.csv").write_text(
        "date,marche,symbole,signal,regime,vol,poids,lots_actuels,lots_cibles,action,resultat\n"
        "2026-10-10,EURUSD,EURUSD,-1.0,0,0.042,-0.53,0.0,-0.04,vente 0.04,ÉCHEC 0 @ None x\n"
        "2026-10-10,NAS100,US Tech 100,1.0,,0.15,0.14,0.0,0.0,aucune,"
        "\"sous le lot minimum (équité nécessaire ≈ 20,000)\"\n", encoding="utf-8")
    (tmp_path / "trend_state.json").write_text(json.dumps(
        {"peak": 10000.0, "equity": 9500.0, "tripped": False, "mode": "réel"}))
    (tmp_path / "trend_equity.csv").write_text(
        "horodatage_utc,jour,equite,exposition,levier_brut,mode\n"
        "2026-10-11T07:00:00+00:00,2026-10-10,9500.0,1.0,1.2,réel\n", encoding="utf-8")
    at = _lancer(monkeypatch, tmp_path)
    assert not at.exception
    assert [m.value for m in at.metric][2] == "5.0%"             # drawdown réel, pas inventé
    assert "NAS100" in at.warning[0].value and "refusé" in at.error[0].value
