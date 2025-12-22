# iXOMA — Intelligent eXtreme Optimization Multi-Agent AI Trader

Multi-agent autonomous trading MVP for WEEX perpetual futures (BTC/USDT, ETH/USDT). Built for the **WEEX hackathon "AI Wars: WEEX Alpha Awakens"**. Demonstrates LLM-driven reasoning, collaborative agents, risk-aware execution, and transparent logging.

## Features
- CrewAI-based multi-agent system (Market Analyzer, Signal Generator, Risk Manager, Execution Agent).
- LLM reasoning (Grok/OpenAI/Gemini/Claude placeholders) with full reasoning logs.
- WEEX integration via `ccxt` (market data, order placement) with **dry-run default**.
- Technical analysis (EMA crossover, RSI-like momentum) and optional sentiment hook.
- Position sizing & risk checks before execution.
- `.env`-driven configuration.
- Simple backtest example for logic sanity checks.
- Ready for hackathon demo: one-command run, verbose logs, clear AI decisions.

## Quickstart
1) Install Python 3.10+.
2) Create and fill `.env` (see `.env.example`).
3) Install deps:
   ```bash
   pip install -r requirements.txt