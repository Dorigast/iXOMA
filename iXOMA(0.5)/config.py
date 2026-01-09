import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict

from dotenv import load_dotenv

ENV_PATH = pathlib.Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)


@dataclass
class Settings:
    env_path: pathlib.Path = field(default=ENV_PATH)

    # --- API / auth ---
    api_key: str = os.getenv("WEEX_API_KEY", "")
    api_secret: str = os.getenv("WEEX_API_SECRET", "")
    api_passphrase: str = os.getenv("WEEX_PASSPHRASE", "")
    base_url: str = os.getenv("WEEX_BASE_URL", "https://api-contract.weex.com")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    symbols: list[str] = field(
        default_factory=lambda: [
            s.strip()
            for s in os.getenv(
                "WEEX_SYMBOLS",
                "cmt_adausdt,cmt_solusdt,cmt_ltcusdt,cmt_dogeusdt,cmt_btcusdt,cmt_ethusdt,cmt_xrpusdt,cmt_bnbusdt",
            ).split(",")
            if s.strip()
        ]
    )
    max_leverage: int = int(os.getenv("WEEX_MAX_LEVERAGE", "5"))
    order_usdt: float = float(os.getenv("WEEX_ORDER_USDT", "10"))
    dry_run: bool = os.getenv("WEEX_DRY_RUN", "false").lower() == "true"
    force_entry: bool = os.getenv("FORCE_ENTRY", "false").lower() == "true"
    fake_weex_mode: bool = os.getenv("WEEX_FAKE_MODE", "false").lower() == "true"

    # --- Risk and timing (P&L percentages are ROE, not raw move) ---
    max_open_positions: int = int(os.getenv("MAX_OPEN_POSITIONS", "8"))
    take_profit_pct: float = float(os.getenv("TAKE_PROFIT_PCT", "0.25"))  # 25% ROE ~= 5% move on 5x
    stop_loss_pct: float = float(os.getenv("STOP_LOSS_PCT", "0.20"))      # 20% ROE ~= 4% move on 5x
    min_close_profit_pct: float = float(os.getenv("MIN_CLOSE_PROFIT_PCT", "0.10"))
    trailing_start_pct: float = float(os.getenv("TRAILING_START_PCT", "0.10"))
    trailing_step_pct: float = float(os.getenv("TRAILING_STEP_PCT", "0.05"))
    stagnation_minutes: int = int(os.getenv("STAGNATION_MINUTES", "30"))
    stagnation_move_pct: float = float(os.getenv("STAGNATION_MOVE_PCT", "0.002"))  # 0.2% move threshold

    candle_granularity: str = os.getenv("WEEX_CANDLE_GRANULARITY", "5m")
    poll_interval_seconds: int = int(os.getenv("WEEX_POLL_INTERVAL", "300"))  # 5 minutes
    signal_threshold: float = float(os.getenv("SIGNAL_THRESHOLD", "0.55"))

    log_level: str = os.getenv("WEEX_LOG_LEVEL", "INFO")

    # --- Feature flags ---
    use_ema_filter: bool = os.getenv("USE_EMA_FILTER", "true").lower() == "true"
    use_ma_filter: bool = os.getenv("USE_MA_FILTER", "true").lower() == "true"
    use_volume_filter: bool = os.getenv("USE_VOLUME_FILTER", "true").lower() == "true"

    # --- Telegram ---
    telegram_enabled: bool = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- Optional overrides for manual fixed sizing (fallback to order_usdt otherwise) ---
    fixed_order_sizes: Dict[str, str] = field(
        default_factory=lambda: {
            "cmt_btcusdt": "0.0002",
            "cmt_ethusdt": "0.01",
            "cmt_solusdt": "0.1",
            "cmt_bnbusdt": "0.1",
            "cmt_xrpusdt": "10",
            "cmt_adausdt": "10",
            "cmt_dogeusdt": "150",
            "cmt_ltcusdt": "0.1",
        }
    )

settings = Settings()
