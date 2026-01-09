import requests
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from config import settings

logger = logging.getLogger("weex_bot")

@dataclass
class TelegramConfig:
    token: str
    chat_id: Optional[str] = None

    @staticmethod
    def from_settings() -> "TelegramConfig":
        return TelegramConfig(
            token=settings.telegram_bot_token or "",
            chat_id=settings.telegram_chat_id or None,
        )

class TelegramNotifier:
    def __init__(self, config: TelegramConfig):
        self.token = config.token
        self.chat_id = config.chat_id
        self.enabled = bool(self.token) and bool(settings.telegram_enabled)

    def send(self, message: str, chat_id: Optional[int | str] = None):
        if not self.enabled:
            return
        cid = chat_id or self.chat_id
        if not cid:
            return
        
        # БЕЗОПАСНАЯ ОТПРАВКА
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            # Таймаут всего 3 секунды. Не успел - пропускаем.
            requests.post(url, json={"chat_id": cid, "text": message}, timeout=3)
        except Exception as e:
            # Просто пишем в лог, но НЕ РОНЯЕМ программу
            logger.warning(f"Telegram send failed (skipping): {e}")

    def get_updates(self, offset: Optional[int] = None, timeout: int = 0) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params: Dict[str, Any] = {}
            if offset is not None:
                params["offset"] = offset
            # Таймаут маленький, чтобы не блокировать цикл
            if timeout:
                params["timeout"] = timeout
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            return data.get("result", [])
        except Exception:
            return []