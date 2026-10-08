"""
Module Telegram - notification par bot Telegram.

Utilise aiohttp. No-op si non configur.
"""
from __future__ import annotations

import logging
import os
import aiohttp

logger = logging.getLogger(__name__)

class TelegramNotifier:
    "Envoie des messages Telegram."
    def __init__(self, token, chat_id):
        self.tg_token = token
        self.tg_chat = chat_id

    @property
    def active(self):
        return bool(
            self.tg_token and self.tg_chat
            and "TON_" not in self.tg_token
            and "TON_" not in self.tg_chat
        )

    async def send(self, msg):
        if not self.active:
            logger.info("[Telegram desactive] " + msg[:120])
            return
        url = "https://api.telegram.org/bot" + self.tg_token + "/sendMessage"
        payload = dict(chat_id=self.tg_chat, text=msg, parse_mode="Markdown")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=15) as resp:
                    if resp.status != 200:
                        payload.pop("parse_mode", None)
                        async with session.post(url, json=payload, timeout=15) as r2:
                            logger.info("Telegram retry status : " + str(r2.status))
        except Exception:
            logger.warning("Telegram : envoi echoue token masque")
