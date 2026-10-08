"""
Client et connexion WebSocket Deriv via le SDK officiel deriv-sdk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import webbrowser
from typing import Any, Optional
from deriv.constants import Config

from deriv_sdk import DerivClient as SDKDerivClient

logger = logging.getLogger(__name__)

OTP_AUTH_URL = (
    "https://oauth2.deriv.com/authorize"
    "?response_type=token"
    "&client_id={app_id}"
    "&redirect_uri={redirect_uri}"
)
DEFAULT_REDIRECT_URI = "https://bot.deriv.com/callback"
_FRAGMENT_TOKEN_RE = re.compile(r"access_token=([^&\s]+)")


class BalanceValue(float):
    @property
    def balance(self):
        return self


class BalanceWrapper:
    def __init__(self, val: float, currency: str = "USD", loginid: str = ""):
        self.balance = BalanceValue(val)
        self.currency = currency
        self.loginid = loginid

    def __float__(self) -> float:
        return float(self.balance)

    def __repr__(self) -> str:
        return f"BalanceWrapper(balance={float(self.balance)}, currency={self.currency!r})"


class ProposalWrapper:
    def __init__(self, proposal_obj: Any):
        self._obj = proposal_obj
        self.proposal = self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._obj, name)

    def __repr__(self) -> str:
        return repr(self._obj)


class DerivTradingProxy:
    def __init__(self, client: SDKDerivClient):
        self._client = client

    async def balance(self) -> BalanceWrapper:
        try:
            res = await self._client.balance.get()
        except Exception as exc:
            logger.error("Deriv balance failed: %s", exc)
            return None
        return BalanceWrapper(
            val=float(res.balance),
            currency=getattr(res, "currency", "USD"),
            loginid=getattr(res, "loginid", ""),
        )

    async def proposal(
        self,
        contract_type: str,
        symbol: str,
        amount: float,
        basis: str = "stake",
        duration: int = 60,
        duration_unit: str = "s",
        currency: str = "USD",
        **kwargs: Any,
    ) -> ProposalWrapper:
        try:
            res = await self._client.proposal.request(
                contract_type=contract_type,
                symbol=symbol,
                amount=amount,
                basis=basis,
                duration=duration,
                duration_unit=duration_unit,
                currency=currency,
                **kwargs,
            )
        except Exception as exc:
            logger.error("Deriv proposal failed: %s", exc)
            return None
        return ProposalWrapper(res)

    async def buy(
        self, buy: str | None = None, proposal_id: str | None = None, price: float = 0.0, **kwargs: Any
    ) -> Any:
        target_id = proposal_id or buy
        if not target_id:
            raise ValueError("ID de proposal manquant pour l'achat Deriv.")
        try:
            return await self._client.buy.buy(proposal_id=str(target_id), price=float(price))
        except Exception as exc:
            logger.error("Deriv buy failed: %s", exc)
            return None


def is_virtual_loginid(loginid: str | None) -> bool:
    """Comptes démo Deriv : loginid VRTC… (VRW… pour les wallets virtuels)."""
    return bool(loginid) and str(loginid).upper().startswith("VR")


class DerivConnection:
    """Wrapper autour du SDK officiel Deriv."""

    def __init__(
        self,
        app_id: Optional[str] = None,
        api_token: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        auto_open_browser: bool = False,
    ):
        raw_token = api_token or Config.API_TOKEN
        self.token = raw_token if raw_token and raw_token != "REMPLACE_PAR_TON_TOKEN" else None
        self.app_id = app_id or Config.APP_ID
        self.redirect_uri = redirect_uri or os.getenv("DERIV_REDIRECT_URI", DEFAULT_REDIRECT_URI)
        self.auto_open_browser = auto_open_browser or os.getenv("DERIV_AUTO_OPEN_BROWSER", "0") == "1"

        self.client = SDKDerivClient(
            app_id=self.app_id,
            api_token=self.token,
        )
        self.client.trading = DerivTradingProxy(self.client)
        if not hasattr(self.client.market, "tick_history"):
            self.client.market.tick_history = self.client.market.ticks_history

        self.is_demo = Config.ACCOUNT_TYPE == "demo"

    @property
    def otp_auth_url(self) -> str:
        from urllib.parse import quote
        return OTP_AUTH_URL.format(
            app_id=quote(self.app_id),
            redirect_uri=quote(self.redirect_uri),
        )

    @staticmethod
    def extract_token_from_url(url):
        pattern = _FRAGMENT_TOKEN_RE
        match = pattern.search(url)
        if match:
            return match.group(1)
        return None

    def _validate_auth(self):
        if not self.client.authorized:
            raise RuntimeError("Autorisation Deriv echouee")
        resp = self.client.auth.authorize_response
        if resp and isinstance(resp, dict) and resp.get("error"):
            err = resp["error"]
            msg = err.get("message", "Erreur inconnue") if isinstance(err, dict) else str(err)
            raise RuntimeError("Autorisation rechassee : " + msg)

    async def _connect_via_otp_url(self):
        auth_url = self.otp_auth_url
        print("========================")
        print("FLUX OTP-URL DERIV")
        print("========================")
        print("URL: " + auth_url)
        print("Redirect URI: " + self.redirect_uri)
        print()
        if self.auto_open_browser:
            try: webbrowser.open(auth_url)
            except Exception as exc: logger.warning("Browser: %s", exc)
        token = await self._read_otp_token_interactive()
        if token:
            self.token = token
            self.client = SDKDerivClient(app_id=self.app_id, api_token=self.token)
            self.client.trading = DerivTradingProxy(self.client)
            logger.info("Token OTP OK")
        else:
            raise RuntimeError("No OTP token")

    async def _read_otp_token_interactive(self):
        tf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".otp_token")
        if os.path.isfile(tf):
            try:
                with open(tf, "r", encoding="utf-8") as f: t = f.read().strip()
                if t: return t
            except Exception as exc:
                logger.warning("OTP token file read failed: %s", exc)
        et = os.getenv("DERIV_OTP_TOKEN", "")
        if et: return et
        try: t = input(" Token > ").strip()
        except (EOFError, KeyboardInterrupt): t = None
        return t if t else None

    async def connect(self):
        if not self.token:
            await self._connect_via_otp_url()

        await self.client.start()
        try:
            self._validate_auth()
        except RuntimeError as exc:
            logger.error("Echec de lauteurisation : %s", exc)
            raise

        account = Config.ACCOUNT_TYPE
        logger.info("Deriv connecte - mode %s", account.upper())

    async def close(self):
        try:
            await self.client.close()
        except Exception as exc:
            logger.error("Deriv close failed: %s", exc)

    async def ping(self) -> dict:
        try:
            return await self.client.request_engine.send({"ping": 1})
        except Exception as exc:
            logger.error("Deriv ping failed: %s", exc)
            return {}

    async def server_time(self) -> float | None:
        """Epoch du serveur Deriv (None si indisponible)."""
        try:
            res = await self.client.request_engine.send({"time": 1})
        except Exception as exc:
            logger.warning("Heure serveur Deriv indisponible : %s", exc)
            return None
        t = res.get("time") if isinstance(res, dict) else getattr(res, "time", None)
        try:
            return float(t)
        except (TypeError, ValueError):
            return None

    async def fetch_loginid(self) -> str | None:
        """loginid du compte RÉELLEMENT autorisé par le token (None si inconnu)."""
        try:
            balance = await self.client.trading.balance()
        except Exception as exc:
            logger.error("Lecture du compte Deriv impossible : %s", exc)
            return None
        loginid = getattr(balance, "loginid", "") if balance is not None else ""
        return str(loginid) or None

    async def get_balance(self) -> float:
        balance = await self.client.trading.balance()
        if balance is None:
            return 0.0
        return float(balance.balance.balance)

    async def get_ticks(self, symbol: str = "R_75") -> dict:
        """Donnees temps reel pour R_75 (Volatility 75)."""
        try:
            history = await self.client.market.tick_history(
                symbol=symbol, count=100, style="candles")
        except Exception as exc:
            logger.error("Deriv ticks failed: %s", exc)
            return {}
        return history

    async def buy_contract(
        self, symbol: str, contract_type: str, amount: float, duration: int
    ) -> dict:
        """Place un contrat."""
        max_auto = float(Config.AUTO_MAX_STAKE)
        if amount > max_auto:
            raise ValueError(
                f"Montant {amount}$ > max auto {max_auto}$. "
                "Confirme manuellement via Telegram."
            )

        proposal = await self.client.trading.proposal(
            contract_type=contract_type,
            symbol=symbol,
            amount=amount,
            basis="stake",
            duration=duration,
            duration_unit="s",
            currency="USD",
        )
        if proposal is None:
            return {}

        buy = await self.client.trading.buy(
            buy=proposal.proposal.id, price=proposal.proposal.ask_price
        )

        logger.info(f"Contrat achete : {contract_type} {symbol} {amount}$ {duration}s")
        return buy

    async def proposal(self, parameters: dict[str, Any]) -> Any:
        """Passerelle pour deriv/bot.py."""
        return await self.client.trading.proposal(**parameters)


DerivClient = DerivConnection
