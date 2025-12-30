import requests
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from config import settings

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
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            requests.post(url, json={"chat_id": cid, "text": message}, timeout=5)
        except Exception:
            pass

    def get_updates(self, offset: Optional[int] = None, timeout: int = 0) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            url = f"https://api.telegram.org/bot{self.token}/getUpdates"
            params: Dict[str, Any] = {}
            if offset is not None:
                params["offset"] = offset
            if timeout:
                params["timeout"] = timeout
            resp = requests.get(url, params=params, timeout=timeout + 5)
            data = resp.json()
            return data.get("result", [])
        except Exception:
            return []
