import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from config import settings


@dataclass
class AISignal:
    side: Optional[str]  # "buy" / "sell" / None
    confidence: float
    reason: str
    regime: str = "llm_scalper"
    suggested_tp_pct: Optional[float] = None
    suggested_sl_pct: Optional[float] = None


def _build_client() -> Optional[OpenAI]:
    if not settings.deepseek_api_key:
        return None
    try:
        return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    except Exception:
        return None


_CLIENT = _build_client()


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["rsi"] = _calculate_rsi(df["close"], period=7)
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    df["atr"] = _calculate_atr(df)
    return df


def _calculate_rsi(series: pd.Series, period: int = 7) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _fallback_signal(df: pd.DataFrame) -> AISignal:
    last = df.iloc[-1]
    price = float(last["close"])
    rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])

    trend_up = ema_fast > ema_slow
    trend_down = ema_fast < ema_slow

    side = None
    reason_parts = [f"RSI {rsi:.1f}", f"EMA9 {ema_fast:.4f}", f"EMA21 {ema_slow:.4f}"]
    if rsi < 32 and trend_up:
        side = "buy"
        reason_parts.append("oversold + uptrend")
    elif rsi > 68 and trend_down:
        side = "sell"
        reason_parts.append("overbought + downtrend")
    else:
        side = None
        reason_parts.append("hold (neutral)")

    confidence = 0.65 if side else 0.4
    return AISignal(side, confidence, "; ".join(reason_parts))


def get_ai_signal(symbol: str, candles_data) -> AISignal:
    """
    Builds a compact context and lets DeepSeek pick LONG/SHORT/HOLD.
    Falls back to a rules-based signal if the key is absent or the call fails.
    """
    try:
        if not isinstance(candles_data, list):
            return AISignal(None, 0.0, "no candle data")
        df = pd.DataFrame(
            candles_data,
            columns=["time", "open", "high", "low", "close", "vol", "amt"],
        )
    except Exception:
        return AISignal(None, 0.0, "data parse error")

    df = _calc_indicators(df)
    last = df.iloc[-1]
    price = float(last["close"])
    rsi = float(last["rsi"]) if not pd.isna(last["rsi"]) else 50.0
    ema_fast = float(last["ema_fast"])
    ema_slow = float(last["ema_slow"])
    atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
    recent_change_pct = ((df["close"].iloc[-1] - df["close"].iloc[-20]) / df["close"].iloc[-20]) * 100 if len(df) >= 20 else 0.0

    if _CLIENT is None:
        return _fallback_signal(df)

    prompt = f"""
You are a short-term futures trader using 5x leverage. Decide the next action for {symbol}.

Context (latest candle):
- Price: {price:.6f}
- RSI(7): {rsi:.2f}
- EMA9: {ema_fast:.6f}, EMA21: {ema_slow:.6f}
- ATR(14): {atr:.6f}
- 20-candle change: {recent_change_pct:.2f}%

Rules:
- Prefer LONG if trend up and momentum not overbought; prefer SHORT if trend down and not oversold.
- Avoid churn: return HOLD if confidence is low.
- Keep TP between 10-35% ROE and SL between 5-25% ROE.

Return JSON with:
{{
  "action": "LONG" | "SHORT" | "HOLD",
  "confidence": 0.0-1.0,
  "reason": "short sentence",
  "tp_pct": 0.10-0.35,
  "sl_pct": 0.05-0.25
}}
"""

    try:
        response = _CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(response.choices[0].message.content)
        action = str(data.get("action", "")).upper()
        side = "buy" if action == "LONG" else ("sell" if action == "SHORT" else None)
        confidence = float(data.get("confidence", 0.0))
        tp_pct = float(data.get("tp_pct", settings.take_profit_pct))
        sl_pct = float(data.get("sl_pct", settings.stop_loss_pct))

        # clamp
        confidence = max(0.0, min(1.0, confidence))
        tp_pct = max(0.05, min(0.40, tp_pct))
        sl_pct = max(0.05, min(0.30, sl_pct))

        reason = data.get("reason") or f"LLM {action}"
        return AISignal(side, confidence, reason, suggested_tp_pct=tp_pct, suggested_sl_pct=sl_pct)
    except Exception as exc:
        return AISignal(None, 0.0, f"AI error: {exc}")


def get_exit_decision(
    symbol: str,
    side: str,
    entry_price: float,
    current_price: float,
    roe_percent: float,
    minutes_open: float,
) -> str:
    """
    Ask the model whether to CLOSE, HOLD, or FLIP based on live PnL and time in trade.
    Falls back to deterministic rules on errors or missing key.
    """
    if _CLIENT is None:
        return _fallback_exit(side, roe_percent)

    prompt = f"""
Manage an existing futures position.
Symbol: {symbol}
Side: {side}
Entry: {entry_price:.6f}
Current: {current_price:.6f}
ROE: {roe_percent:+.2f}%
Minutes open: {minutes_open:.1f}

Guidance:
- Close if loss exceeds stop or if profit target hit.
- Flip only if confidence of reversal is very high.
- Otherwise hold with a tighter stop if profit is decent.

Return JSON: {{"action": "CLOSE" | "HOLD" | "FLIP", "note": "short reason"}}
"""
    try:
        response = _CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        data = json.loads(response.choices[0].message.content)
        action = str(data.get("action", "HOLD")).upper()
        return action if action in {"CLOSE", "HOLD", "FLIP"} else "HOLD"
    except Exception:
        return _fallback_exit(side, roe_percent)


def _fallback_exit(side: str, roe_percent: float) -> str:
    if roe_percent >= settings.take_profit_pct * 100:
        return "CLOSE"
    if roe_percent <= -settings.stop_loss_pct * 100:
        return "CLOSE"
    if abs(roe_percent) < settings.min_close_profit_pct * 100:
        return "HOLD"
    return "HOLD"
