"""main.py doit lancer le vrai bot, en dry-run par défaut."""
import asyncio

import pandas as pd
import pytest

import main as entry
from deriv.bot_executor import DerivBotExecutor


@pytest.fixture
def appels(monkeypatch):
    calls = {}

    async def fake_executor_main(dry_run):
        calls["dry_run"] = dry_run

    monkeypatch.setattr("deriv.bot_executor.main", fake_executor_main)
    monkeypatch.setattr(entry.subprocess, "Popen", lambda *a, **k: pytest.fail("dashboard lancé"))
    return calls


def test_deriv_est_en_dry_run_par_defaut(appels):
    entry.main(["deriv"])
    assert appels["dry_run"] is True


def test_live_desactive_le_dry_run(appels):
    entry.main(["deriv", "--live"])
    assert appels["dry_run"] is False


def test_dry_run_n_achete_aucun_contrat(monkeypatch, caplog):
    bot = DerivBotExecutor(capital_usd=150.0, dry_run=True)
    detail = {"HMM": {"regime": "Bull"}, "XGBoost": {"proba": 0.8},
              "LSTM": {"proba": 0.8}, "Kalman": {"trend": 0.1}, "RSI": {"rsi": 40}}
    monkeypatch.setattr(bot.ensemble, "predict", lambda df: {
        "signal": "BUY", "tradeable": True, "confiance": 0.8,
        "buy_votes": 5, "sell_votes": 0, "hold_votes": 2, "detail": detail})

    async def interdit(*a, **k):
        pytest.fail("contrat acheté en dry-run")

    monkeypatch.setattr(bot, "_execute_contract", interdit)
    monkeypatch.setattr(bot, "_journaliser_setup", interdit)
    df = pd.DataFrame({"close": [1.0, 1.1]},
                      index=pd.to_datetime(["2026-01-05 11:00", "2026-01-05 12:00"], utc=True))
    with caplog.at_level("INFO", logger="DERIV_BOT"):
        asyncio.run(bot._run_cycle(df))
    assert "DRY-RUN" in caplog.text  # le cycle est bien allé jusqu'à l'exécution


def test_env_charge_avant_config(tmp_path):
    """Les valeurs du .env doivent atteindre Config (bug : chargé trop tard)."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    (tmp_path / "deriv").mkdir()
    shutil.copy(Path(entry.__file__).parent / "deriv" / "constants.py",
                tmp_path / "deriv" / "constants.py")
    (tmp_path / ".env").write_text("DERIV_SYMBOL=BOOM500\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if k not in ("BOT_SKIP_DOTENV", "DERIV_SYMBOL")}
    out = subprocess.run(
        [sys.executable, "-c",
         "import importlib.util as u;"
         "s=u.spec_from_file_location('c','deriv/constants.py');"
         "m=u.module_from_spec(s);s.loader.exec_module(m);print(m.Config.SYMBOL)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "BOOM500"
