import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

env_path = pathlib.Path(__file__).parent / '.env'
load_dotenv(env_path)

@dataclass
class Settings:
    api_key: str = os.getenv("WEEX_API_KEY", "")
    api_secret: str = os.getenv("WEEX_API_SECRET", "")
    api_passphrase: str = os.getenv("WEEX_PASSPHRASE", "")
    base_url: str = os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com")

    symbols: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv("WEEX_SYMBOLS", "cmt_btcusdt,cmt_ethusdt,cmt_solusdt,cmt_bnbusdt,cmt_xrpusdt").split(",")
            if s.strip()
        ]
    )
    max_leverage: int = int(os.getenv("WEEX_MAX_LEVERAGE", "10"))

    # --- ЖЕСТКИЕ РАЗМЕРЫ ПОЗИЦИЙ (ИСПРАВЛЕНО ДЛЯ BNB) ---
    # BNB изменен с 0.03 на 0.1 из-за требований биржи (stepSize 0.1)
    # Внимание: 0.1 BNB по курсу 850$ = 85$ маржи.
    # Если это много, удалите BNB из списка символов в .env
    fixed_order_sizes: Dict[str, str] = field(
        default_factory=lambda: {
            "cmt_btcusdt": "0.0002", # ~18$
            "cmt_ethusdt": "0.01",   # ~30$
            "cmt_solusdt": "0.1",    # ~20$
            "cmt_bnbusdt": "0.1",    # ~85$ (Минимально возможный лот)
            "cmt_xrpusdt": "10",     # ~25$
        }
    )
    # -------------------------------

    order_usdt: float = float(os.getenv("WEEX_ORDER_USDT", "10")) # Для обратной совместимости
    dry_run: bool = os.getenv("WEEX_DRY_RUN", "false").lower() == "true"
    force_entry: bool = os.getenv("FORCE_ENTRY", "true").lower() == "true"

    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.04"))
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.10"))

    candle_granularity: str = os.getenv("WEEX_CANDLE_GRANULARITY", "5m")
    short_window: int = int(os.getenv("WEEX_SHORT_WINDOW", "10"))
    long_window: int = int(os.getenv("WEEX_LONG_WINDOW", "30"))
    poll_interval_seconds: int = int(os.getenv("WEEX_POLL_INTERVAL", "60"))

    log_level: str = os.getenv("WEEX_LOG_LEVEL", "INFO")

    telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    trailing_start_pct: float = float(os.getenv("TRAILING_START_PCT", "0.01"))
    trailing_step_pct: float = float(os.getenv("TRAILING_STEP_PCT", "0.003"))

    signal_threshold: float = float(os.getenv("SIGNAL_THRESHOLD", "0.90"))

    use_ema_filter: bool = os.getenv("USE_EMA_FILTER", "true").lower() == "true"
    use_ma_filter: bool = os.getenv("USE_MA_FILTER", "true").lower() == "true"
    use_volume_filter: bool = os.getenv("USE_VOLUME_FILTER", "true").lower() == "true"

settings = Settings()