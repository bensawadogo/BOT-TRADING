"""Exécution des contrats : aucun faux trade, chaque trade clôturé avec SON PnL."""
import asyncio
from types import SimpleNamespace

import pytest

from deriv.bot_executor import DerivBotExecutor


class FauxJournal:
    def __init__(self):
        self.exits = []

    def log_exit_contract(self, trade_id, pnl_dollar, notes=""):
        self.exits.append((trade_id, pnl_dollar, notes))
        return {"pnl_net_pct": pnl_dollar}


def _bot(monkeypatch, *, proposal_ok=True, contract_id=123, etats=()):
    """Bot avec une fausse connexion Deriv. `etats` = réponses successives de contract.get."""
    bot = DerivBotExecutor(capital_usd=150.0)
    monkeypatch.setattr(bot, "DURATION", 0)
    monkeypatch.setattr(bot, "SETTLE_POLL_SEC", 0)
    monkeypatch.setattr(bot, "SETTLE_RETRIES", 3)
    bot.journal, bot.learner, bot.online = FauxJournal(), None, None
    reponses = list(etats)

    async def proposal(**k):
        return SimpleNamespace(proposal=SimpleNamespace(id="P1", ask_price=2.0)) if proposal_ok else None

    async def buy(**k):
        return SimpleNamespace(contract_id=contract_id)

    async def get(contract_id):
        return reponses.pop(0) if reponses else SimpleNamespace(is_sold=0, profit=0.0)

    bot.conn = SimpleNamespace(client=SimpleNamespace(
        trading=SimpleNamespace(proposal=proposal, buy=buy),
        contract=SimpleNamespace(get=get)))

    async def journaliser(df, result):
        bot.open_trades["T1"] = {"entry": None}
        return "T1"

    monkeypatch.setattr(bot, "_journaliser_setup", journaliser)
    return bot


def run(bot):
    return asyncio.run(bot._execute_contract("CALL", df=object(), result={}))


def test_achat_refuse_ne_compte_aucun_trade(monkeypatch):
    bot = _bot(monkeypatch, proposal_ok=False)
    assert run(bot) is None
    assert bot.risk.total_trades == 0          # avant : compté comme un gain à 0 $
    assert bot.journal.exits == []


def test_achat_sans_contract_id_ne_compte_aucun_trade(monkeypatch):
    bot = _bot(monkeypatch, contract_id=0)
    assert run(bot) is None
    assert bot.risk.total_trades == 0


def test_attend_le_reglement_et_cloture_son_propre_trade(monkeypatch):
    bot = _bot(monkeypatch, etats=[
        SimpleNamespace(is_sold=0, profit=1.5),   # pas encore réglé : ignoré
        SimpleNamespace(is_sold=1, profit=-2.0),  # réglé : perte
    ])
    assert run(bot) == -2.0
    assert bot.journal.exits == [("T1", -2.0, "")]
    assert bot.risk.total_trades == 1 and bot.risk.consecutive_loss == 1
    assert bot.open_trades == {}


def test_resultat_inconnu_compte_comme_perte(monkeypatch):
    bot = _bot(monkeypatch, etats=[])          # jamais réglé
    assert run(bot) is None
    assert bot.risk.consecutive_loss == 1      # prudence : perte de la mise
    trade_id, pnl, notes = bot.journal.exits[0]
    assert trade_id == "T1" and pnl == pytest.approx(-bot.STAKE_AUTO) and notes
