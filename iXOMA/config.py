import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    weex_api_key: str = os.getenv("WEEX_API_KEY", "")
    weex_api_secret: str = os.getenv("WEEX_API_SECRET", "")
    weex_password: str = os.getenv("WEEX_PASSWORD", "")

    llm_provider: str = os.getenv("LLM_PROVIDER", "dummy")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    dry_run: bool = os.getenv("DRY_RUN", "true").lower() == "true"
    target_symbols: list[str] = (
        [s.strip() for s in os.getenv("TARGET_SYMBOLS", "BTC/USDT").split(",") if s.strip()]
    )
    max_position_usdt: float = float(os.getenv("MAX_POSITION_USDT", "500"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.02"))
    ta_lookback: int = int(os.getenv("TA_LOOKBACK", "50"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
