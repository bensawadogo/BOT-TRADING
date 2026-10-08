"""
═══════════════════════════════════════════════════════════════════
ÉTAPE 1 — COLLECTE DE DONNÉES DERIV EN TEMPS RÉEL
deriv/data_collector.py
═══════════════════════════════════════════════════════════════════

Collecte les ticks et bougies Deriv via WebSocket officiel (deriv-sdk,
framework standard du projet — python-deriv-api est incompatible car il
verrouille websockets==10.3 alors que deriv-sdk exige websockets>=16).
Construit un DataFrame OHLCV pour alimenter les modèles.

Adapté à la couche existante deriv/client.py (DerivConnection).
"""
from __future__ import annotations

import asyncio
import logging
from deriv.constants import Config

import pandas as pd

logger = logging.getLogger(__name__)


class DerivDataCollector:
    """
    Se connecte à Deriv via WebSocket et collecte les ticks en temps réel.
    Construit des bougies OHLCV toutes les N secondes (bucket non chevauchant).

    Deux sources de données :
      1. subscribe_ticks()   → flux temps réel (Subscription[Tick] du SDK)
      2. get_candle_history()→ historique server-side (bougies OHLCV)
    """

    APP_ID = Config.APP_ID

    # Symboles recommandés pour l'algo (évite les synthétiques purs)
    SYMBOLES = {
        "R_75": "Volatility 75 — synthétique",
        "R_50": "Volatility 50 — moins volatile",
        "frxEURUSD": "EUR/USD — vrai marché Forex",
        "cryBTCUSD": "BTC/USD — crypto via Deriv",
    }

    def __init__(self, symbol: str = "frxEURUSD", tick_size: int = 60,
                 connection=None):
        self.symbol = symbol
        self.tick_size = max(int(tick_size), 1)  # secondes par bougie
        self.ticks: list[dict] = []
        self.candles: list[dict] = []
        self.conn = connection
        self._own_connection = connection is None
        self._connected = False
        self._subscription = None
        self._last_bucket = None
        self.api = None

    # ── Connexion WebSocket ──────────────────────────────────────────
    async def connect(self):
        """Ouvre la connexion WebSocket (partagée possible avec le bot)."""
        if self.conn is None:
            from deriv.client import DerivConnection  # import différé
            self.conn = DerivConnection(
                app_id=self.APP_ID,
                api_token=Config.API_TOKEN,
            )
            self._own_connection = True

        started = (
            getattr(self.conn, "client", None) is not None
            and getattr(self.conn.client, "started", False)
        )
        if not started:
            await self.conn.connect()

        self._connected = True
        self.api = getattr(self.conn, "client", None)
        logger.info(f"Connecté à Deriv — symbole : {self.symbol}")
        return self

    async def close(self):
        """Ferme l'abonnement et la connexion (si elle nous appartient)."""
        if getattr(self, "_subscription", None) is not None:
            try:
                close = getattr(self._subscription, "close", None)
                if callable(close):
                    await close()
            except Exception as exc:
                logger.warning('Close cleanup failed: %s', exc)
            self._subscription = None
        if self._own_connection and self.conn is not None:
            try:
                await self.conn.close()
            except Exception as exc:
                logger.warning('Conn close failed: %s', exc)
            self.conn = None
        self._connected = False

    # ── Flux temps réel ──────────────────────────────────────────────
    async def subscribe_ticks(self):
        """
        S'abonne aux ticks et construit les bougies en continu.
        Boucle infinie : se termine si le flux est coupé.
        """
        if not self._connected:
            await self.connect()

        self._subscription = await self.conn.client.market.subscribe_ticks(
            self.symbol
        )
        logger.info(f"Abonnement ticks actif — {self.symbol}")

        try:
            async for tick in self._subscription:
                if tick is None:
                    continue
                self.process_tick(int(tick.epoch), float(tick.quote))
        finally:
            self._subscription = None
            logger.warning("Flux de ticks terminé (reconnexion nécessaire)")

    def process_tick(self, epoch: int, price: float):
        """Reçoit un tick brut (SDK ou tests) et construit les bougies."""
        self.ticks.append({"time": int(epoch), "price": float(price)})
        self._build_candle()

    # ── Agrégation OHLCV ─────────────────────────────────────────────
    def _build_candle(self):
        """
        Agrége les ticks en bougie OHLCV toutes les tick_size secondes.
        Un bucket est fermé dès qu'un tick franchit une nouvelle frontière
        de tick_size secondes → bougies non chevauchantes et stables.
        """
        if len(self.ticks) < 2:
            return

        now = self.ticks[-1]["time"]
        bucket = now // self.tick_size

        # Une seule bougie par bucket (fermeture immuable ensuite)
        if self._last_bucket is not None and bucket == self._last_bucket:
            return
        self._last_bucket = bucket

        window_end = bucket * self.tick_size
        window_start = window_end - self.tick_size

        window_ticks = [
            t for t in self.ticks
            if window_start <= t["time"] < window_end
        ]
        # Fallback si trou de données (marché fermé) : derniers ticks connus
        if len(window_ticks) < 2:
            window_ticks = [
                t for t in self.ticks if t["time"] >= window_end - self.tick_size
            ]
        if not window_ticks:
            return

        prices = [t["price"] for t in window_ticks]
        self.candles.append({
            "timestamp": window_end,
            "open":   prices[0],
            "high":   max(prices),
            "low":    min(prices),
            "close":  prices[-1],
            "volume": len(prices),  # nb de ticks = proxy du volume
        })

        # Garde seulement les 500 dernières bougies en mémoire
        if len(self.candles) > 500:
            self.candles = self.candles[-500:]
        # Idem pour les ticks bruts (mémoire bornée)
        if len(self.ticks) > 5000:
            self.ticks = self.ticks[-5000:]

    def get_dataframe(self, min_candles: int = 50) -> pd.DataFrame:
        """Retourne le DataFrame OHLCV des bougies (vide si trop peu)."""
        if len(self.candles) < min_candles:
            return pd.DataFrame()
        df = pd.DataFrame(self.candles)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        return df

    # ── Historique server-side ───────────────────────────────────────
    async def get_candle_history(self, count: int = 500) -> pd.DataFrame:
        """
        Récupère l'historique des bougies depuis l'API Deriv
        (ticks_history style=candles, granularity = tick_size secondes).
        """
        if not self._connected:
            await self.connect()

        # API v4 :
        # - granularity accepte tout entier
        # - subscribe=0 : requête ponctuelle (omis du message WS car Deriv rejette subscribe=0)
        # - adjust_start_time=1 : alignement des bougies
        payload = {
            "ticks_history": self.symbol,
            "count": count,
            "end": "latest",
            "start": 1,
            "style": "candles",
            "granularity": self.tick_size,  # accepte tout entier (API v4)
            "adjust_start_time": 1,
        }

        if hasattr(self.conn.client, "market") and hasattr(self.conn.client.market, "request"):
            history = await self.conn.client.market.request(payload, expected="candles")
        else:
            history = await self.conn.client.request_engine.send(payload, expected_msg_type="candles")

        candles = history.get("candles", []) or []
        if not candles:
            return pd.DataFrame()

        # API v4 : les bougies peuvent être des dicts ou des objets
        def _candle_field(c, field, default=0.0):
            if isinstance(c, dict):
                return c.get(field, default)
            return getattr(c, field, default)

        rows = [{
            "timestamp": _candle_field(c, "epoch", _candle_field(c, "time")),
            "open": float(_candle_field(c, "open")),
            "high": float(_candle_field(c, "high")),
            "low": float(_candle_field(c, "low")),
            "close": float(_candle_field(c, "close")),
            "volume": 1,  # Deriv ne donne pas le volume réel
        } for c in candles]

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        logger.info(f"Historique chargé : {len(df)} bougies ({self.symbol})")
        return df

    async def _send_candle_request(self, payload: dict) -> dict:
        """Envoie une requête de bougies à l'API Deriv v4."""
        if hasattr(self, "api") and self.api is not None and hasattr(self.api, "send"):
            try:
                return await self.api.send(payload)
            except Exception as exc:
                logger.warning('CCollector error: %s', exc)
        if self.conn and getattr(self.conn, "client", None) is not None:
            client = self.conn.client
            if hasattr(client, "market") and hasattr(client.market, "request"):
                return await client.market.request(payload, expected="candles")
            if hasattr(client, "request_engine") and hasattr(client.request_engine, "send"):
                return await client.request_engine.send(payload, expected_msg_type="candles")
        raise RuntimeError("Non connecté à Deriv")

    async def get_candle_history_full(
        self, target_count: int = 5000
    ) -> pd.DataFrame:
        """
        Charge l'historique complet par pagination epoch.
        Fait N requêtes de 1000 bougies max en reculant le `end`.
        Deriv ne documente pas de limite sur la profondeur historique.
        """
        if not self._connected:
            await self.connect()

        all_candles: list[dict] = []
        end_val: str = "latest"          # commence par 'latest' pour avoir les données les plus récentes
        chunk_size = 1000                # max par requête
        last_seen_epoch = None

        while len(all_candles) < target_count:
            nb_a_demander = min(chunk_size, target_count - len(all_candles))
            payload = {
                "ticks_history": self.symbol,
                "count": nb_a_demander,
                "end": end_val,
                "style": "candles",
                "granularity": self.tick_size,
                "adjust_start_time": 1,
            }

            response = await self._send_candle_request(payload)
            raw_candles = response.get("candles", []) or []
            if not raw_candles:
                break  # plus de données disponibles

            candles = []
            for c in raw_candles:
                if isinstance(c, dict):
                    candles.append(c)
                else:
                    candles.append({
                        "epoch": getattr(c, "epoch", getattr(c, "time", 0)),
                        "open": float(getattr(c, "open", 0.0)),
                        "high": float(getattr(c, "high", 0.0)),
                        "low": float(getattr(c, "low", 0.0)),
                        "close": float(getattr(c, "close", 0.0)),
                    })

            all_candles = candles + all_candles  # prepend (ordre chronologique)

            # Recule le end au début de ce chunk
            first_epoch = candles[0].get("epoch", candles[0].get("time", None))
            if first_epoch is None or first_epoch == last_seen_epoch:
                break  # plus de progression temporelle
            last_seen_epoch = first_epoch
            end_val = str(int(first_epoch) - 1)

            logger.info(
                f"Chunk chargé : {len(candles)} bougies | "
                f"Total : {len(all_candles)} | "
                f"Début chunk : {pd.to_datetime(int(first_epoch), unit='s').date()}"
            )

            await asyncio.sleep(0.5)  # rate limit

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(all_candles[-target_count:])
        epoch_col = "epoch" if "epoch" in df.columns else ("time" if "time" in df.columns else "timestamp")
        df[epoch_col] = pd.to_datetime(df[epoch_col], unit="s")
        df.rename(columns={epoch_col: "timestamp"}, inplace=True)
        df.drop_duplicates(subset=["timestamp"], inplace=True)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].astype(float)
        df["volume"] = 1.0

        logger.info(f"Historique complet chargé : {len(df)} bougies ({self.symbol})")
        return df