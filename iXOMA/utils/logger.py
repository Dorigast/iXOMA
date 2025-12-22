import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

def setup_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("ixoma")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = RotatingFileHandler(LOG_PATH / "ixoma.log", maxBytes=5_000_000, backupCount=3)
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger

def log_reasoning(logger: logging.Logger, agent: str, message: str):
    logger.info(f"[REASONING] {agent}: {message}")
