import os
import random
import numpy as np
import pandas as pd
from crewai import Crew, Process
from config import settings
from utils.logger import setup_logger
from weex_api import WeexAPI
from tools.weex_tools import make_tools, tool_descriptions
from agents.market_analyzer import build_market_analyzer
from agents.signal_generator import build_signal_generator
from agents.risk_manager import build_risk_manager
from agents.execution_agent import build_execution_agent
from tasks.analyze_market import task_analyze_market
from tasks.generate_signal import task_generate_signal
from tasks.manage_risk import task_manage_risk
from tasks.execute_trade import task_execute_trade

# Placeholder LLM wrapper
class DummyLLM:
    def __call__(self, prompt: str, **kwargs):
        # Simple stub with deterministic but varied responses
        actions = ["LONG", "SHORT", "FLAT"]
        action = random.choice(actions)
        confidence = round(random.uniform(0.4, 0.8), 2)
        return f"{action}|{confidence}|heuristic stub"

def load_llm():
    provider = settings.llm_provider.lower()
    if provider == "dummy" or not settings.llm_api_key:
        return DummyLLM()
    # Placeholder hook: users can integrate specific SDKs here.
    return DummyLLM()

def compute_ta(ohlcv):
    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ema_fast"] = df["close"].ewm(span=10).mean()
    df["ema_slow"] = df["close"].ewm(span=30).mean()
    df["momentum"] = df["close"].pct_change().rolling(5).mean().fillna(0)
    df["signal_strength"] = (df["ema_fast"] - df["ema_slow"]) / df["close"]
    latest = df.iloc[-1]
    return {
        "ema_fast": latest["ema_fast"],
        "ema_slow": latest["ema_slow"],
        "momentum": latest["momentum"],
        "signal_strength": latest["signal_strength"],
    }

def summarize_book(book, depth: int = 3):
    bids = book.get("bids", [])[:depth]
    asks = book.get("asks", [])[:depth]
    return {"bids": bids, "asks": asks}

def run_cycle(symbol: str, logger):
    llm = load_llm()
    api = WeexAPI(dry_run=settings.dry_run)
    tools = make_tools(api)

    analyzer = build_market_analyzer(llm, tools, logger)
    signaler = build_signal_generator(llm, tools, logger)
    risk = build_risk_manager(llm, tools, logger)
    exec_agent = build_execution_agent(llm, tools, logger)

    ticker = api.fetch_ticker(symbol)
    book = api.fetch_order_book(symbol)
    ohlcv = api.fetch_ohlcv(symbol, limit=60)
    ta = compute_ta(ohlcv)
    book_summary = summarize_book(book)

    ta_payload = {
        "ticker": {k: ticker.get(k) for k in ["last", "bid", "ask", "percentage"]},
        "order_book_summary": book_summary,
        **ta,
    }

    config_snapshot = {
        "max_position_usdt": settings.max_position_usdt,
        "risk_per_trade": settings.risk_per_trade,
        "dry_run": settings.dry_run,
    }

    tasks = [
        task_analyze_market(analyzer, symbol, ta_payload),
        task_generate_signal(signaler, symbol),
        task_manage_risk(risk, symbol, config_snapshot),
        task_execute_trade(exec_agent, symbol),
    ]

    crew = Crew(
        agents=[analyzer, signaler, risk, exec_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=2,
    )

    result = crew.kickoff()
    logger.info(f"Final outcome for {symbol}: {result}")

def main():
    logger = setup_logger(settings.log_level)
    logger.info("Starting iXOMA crew (dry-run=%s)", settings.dry_run)
    for sym in settings.target_symbols:
        try:
            run_cycle(sym, logger)
        except Exception as exc:
            logger.exception("Error during cycle for %s: %s", sym, exc)

if __name__ == "__main__":
    main()
