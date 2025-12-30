import logging
import sys

def setup_logger(level: str = "INFO"):
    logger = logging.getLogger("weex_bot")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger

def log_reasoning(logger, title: str, message: str):
    logger.info("\n----------------------------------------\n%s\n%s\n----------------------------------------", title, message)

def log_api_error(logger, context: str, error: str):
    logger.error("API error in %s: %s", context, error)
