from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Signal:
    side: str           # 'buy', 'sell', 'none'
    confidence: float
    reason: str
    regime: str         # 'TREND' | 'RANGE' | 'UNKNOWN'


def _extract_closes(candles: List) -> List[float]:
    closes = []
    for c in candles:
        try:
            if isinstance(c, dict):
                closes.append(float(c.get("close")))
            else:
                closes.append(float(c[4]))
        except Exception:
            continue
    return closes


def moving_average_signal(
    candles: List,
    short_window: int = 10,
    long_window: int = 30,
    threshold: float = 0.9,
    use_ema_filter: bool = True,
    use_ma_filter: bool = True,
    use_volume_filter: bool = False,
) -> Signal:
    closes = _extract_closes(candles)
    if len(closes) < max(short_window, long_window) + 2:
        return Signal(side="none", confidence=0.0, reason="Not enough data", regime="UNKNOWN")

    def sma(values, window):
        if len(values) < window:
            return None
        return sum(values[-window:]) / window

    short_ma_prev = sma(closes[:-1], short_window)
    long_ma_prev = sma(closes[:-1], long_window)
    short_ma_now = sma(closes, short_window)
    long_ma_now = sma(closes, long_window)

    if short_ma_prev is None or long_ma_prev is None or short_ma_now is None or long_ma_now is None:
        return Signal(side="none", confidence=0.0, reason="Not enough data", regime="UNKNOWN")

    if abs(long_ma_now - long_ma_prev) / long_ma_prev if long_ma_prev else 0 > 0.001:
        regime = "TREND"
    else:
        regime = "RANGE"

    cross_up = short_ma_prev <= long_ma_prev and short_ma_now > long_ma_now
    cross_down = short_ma_prev >= long_ma_prev and short_ma_now < long_ma_now

    conf = 0.0
    side = "none"
    reason = "No strong signal"

    if cross_up:
        side = "buy"
        conf = 0.93
        reason = "MA cross up (potential long)"
    elif cross_down:
        side = "sell"
        conf = 0.93
        reason = "MA cross down (exit/short)"

    if use_ma_filter:
        delta = (short_ma_now - long_ma_now) / long_ma_now if long_ma_now else 0.0
        conf += abs(delta) * 5

    if regime == "TREND":
        conf += 0.02

    conf = max(0.0, min(0.99, conf))

    if conf < threshold:
        return Signal(side="none", confidence=conf, reason="No strong signal", regime=regime)

    return Signal(side=side, confidence=conf, reason=reason, regime=regime)
