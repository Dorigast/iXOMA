import argparse
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple

from config import settings
from strategy.ai_strategy import AISignal, get_ai_signal, get_exit_decision
from utils.logger import log_api_error, log_reasoning, setup_logger
from utils.stats import BotStats
from utils.telegram_notifier import TelegramConfig, TelegramNotifier
from weex_api import FakeWeexAPI, WeexAPI, WeexAPIError


@dataclass
class PositionInfo:
    symbol: str
    side: str
    size: float
    entry_price: Optional[float]
    margin: float
    unrealized_pnl: float


@dataclass
class PositionState:
    open_ts: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    stagnation_ref_price: Optional[float] = None
    last_move_ts: Optional[float] = None
    last_signal: Optional[AISignal] = None


TYPE_OPEN_LONG = 1
TYPE_OPEN_SHORT = 2
TYPE_CLOSE_LONG = 3
TYPE_CLOSE_SHORT = 4


def parse_positions_response(raw) -> List[PositionInfo]:
    if isinstance(raw, dict):
        data = raw.get("data") or raw
    elif isinstance(raw, list):
        data = raw
    else:
        data = []

    items: List = []
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        items = data.get("positions") or data.get("list") or []

    result: List[PositionInfo] = []
    for p in items or []:
        if not isinstance(p, dict):
            continue
        symbol = p.get("symbol")
        if not symbol:
            continue
        try:
            size = float(p.get("size") or p.get("positionSize") or p.get("availQty") or 0)
        except Exception:
            size = 0.0
        if size <= 0:
            continue

        side = str(p.get("side") or p.get("positionSide") or "LONG").upper()
        try:
            margin = float(p.get("marginSize") or p.get("margin") or 0)
        except Exception:
            margin = 0.0
        try:
            unreal = float(p.get("unrealizePnl") or p.get("unrealizedPnl") or 0)
        except Exception:
            unreal = 0.0
        entry = None
        try:
            ov = float(p.get("open_value") or p.get("openValue") or 0)
            if ov > 0 and size > 0:
                entry = ov / size
        except Exception:
            entry = None
        result.append(PositionInfo(symbol, side, size, entry, margin, unreal))
    return result


def read_all_positions(api: WeexAPI, logger=None) -> List[PositionInfo]:
    try:
        raw = api.get_positions()
        return parse_positions_response(raw)
    except Exception as exc:
        if logger:
            logger.error("Error reading positions: %s", exc)
        return []


def get_last_price_from_ticker(ticker) -> Optional[float]:
    if not isinstance(ticker, dict):
        return None
    data = ticker.get("data") or ticker
    if not isinstance(data, dict):
        return None
    for key in ("last", "markPrice", "close", "price"):
        v = data.get(key)
        if v in (None, ""):
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def quantize_size(size: float) -> str:
    if size <= 0:
        return "0"
    return str(Decimal(size).quantize(Decimal("0.000001"), rounding=ROUND_DOWN))


def calc_size_for_order(symbol: str, last_price: float) -> Optional[Tuple[str, float]]:
    if last_price <= 0:
        return None
    if settings.order_usdt > 0:
        notional = settings.order_usdt * settings.max_leverage
        size = notional / last_price
        size_str = quantize_size(size)
        return size_str, float(size_str)

    size_str = settings.fixed_order_sizes.get(symbol)
    if size_str:
        try:
            return size_str, float(size_str)
        except Exception:
            return size_str, 0.0
    return None


def calc_roe_pct(pos: PositionInfo, last_price: float) -> float:
    if not pos.entry_price or not last_price:
        return 0.0
    direction = 1.0 if pos.side.startswith("LONG") else -1.0
    return (last_price - pos.entry_price) / pos.entry_price * 100.0 * direction


def format_help() -> str:
    return "\n".join(
        [
            "/help - list commands",
            "/status - positions status",
            "/balance - balance and PnL",
            "/close all - close all positions",
            "/close <coin> - close by coin (btc/eth/etc)",
            "/stop - stop the bot",
        ]
    )


def format_status_text(
    api: WeexAPI,
    symbols: List[str],
    symbol_states: Dict[str, PositionState],
    positions: List[PositionInfo],
) -> str:
    lines: List[str] = ["Market status:"]
    pos_by_symbol = {p.symbol: p for p in positions}

    for sym in symbols:
        pos = pos_by_symbol.get(sym)
        st = symbol_states.get(sym) or PositionState()
        if pos is None:
            lines.append(f"- {sym}: no open position")
            continue
        try:
            price = get_last_price_from_ticker(api.get_ticker(sym)) or pos.entry_price or 0.0
        except Exception:
            price = pos.entry_price or 0.0
        roe = calc_roe_pct(pos, price)
        tp = settings.take_profit_pct * 100
        sl = settings.stop_loss_pct * 100
        lines.append(f"- {sym} ({pos.side}) size={pos.size:.6f}")
        lines.append(f"  Entry {pos.entry_price:.6f} | Last {price:.6f} | ROE {roe:+.2f}%")
        if st.trailing_stop_pct is not None:
            lines.append(f"  Trailing stop at {st.trailing_stop_pct:.2f}%")
        lines.append(f"  TP {tp:.1f}% | SL {sl:.1f}% | Opened {time.ctime(st.open_ts) if st.open_ts else 'n/a'}")
    return "\n".join(lines)


def format_balance_report(api: WeexAPI, positions: List[PositionInfo]) -> str:
    try:
        assets = api.get_assets()
        data = assets.get("data") if isinstance(assets, dict) else assets
        row = data[0] if isinstance(data, list) and data else data if isinstance(data, dict) else None
        if not row:
            return "Balance data is empty."
        balance = float(row.get("balance") or row.get("total_balance") or 0.0)
        equity = float(row.get("equity") or row.get("totalEquity") or row.get("total_equity") or 0.0)
        available = float(row.get("available") or row.get("availableBalance") or row.get("available_balance") or 0.0)
    except Exception as exc:
        return f"Error fetching balance: {exc}"

    total_margin = sum(p.margin for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    pnl_icon = "[OK]" if total_unrealized_pnl >= 0 else "[WARN]"

    lines = [
        "Wallet:",
        f"Equity: {equity:.2f} USDT",
        f"Available: {available:.2f} USDT",
        f"In trades (margin): {total_margin:.2f} USDT",
        f"{pnl_icon} Unrealized PnL: {total_unrealized_pnl:+.2f} USDT",
    ]
    return "\n".join(lines)


def close_position_market(
    api: WeexAPI,
    logger,
    notifier: TelegramNotifier,
    pos: PositionInfo,
    last_price: float,
    chat_id: Optional[str],
    reason: str,
):
    side_upper = pos.side.upper()
    order_side = "sell" if side_upper.startswith("LONG") else "buy"
    type_code = TYPE_CLOSE_LONG if side_upper.startswith("LONG") else TYPE_CLOSE_SHORT
    size_str = quantize_size(pos.size)
    params = dict(
        symbol=pos.symbol,
        client_oid=f"close-{uuid.uuid4().hex[:10]}",
        size=size_str,
        side=order_side,
        type_code=type_code,
        price=None,
        order_type="1",
        match_price="1",
    )
    logger.info("Closing %s at %.6f (%s) params=%s", pos.symbol, last_price, reason, params)
    if settings.dry_run:
        logger.info("[DRY RUN] close skipped")
        return
    try:
        res = api.place_order(**params)
        logger.info("Close response: %s", res)
        if chat_id:
            notifier.send(f"Closed {pos.symbol} {pos.side} size={pos.size:.6f} | reason: {reason}", chat_id=chat_id)
    except WeexAPIError as exc:
        log_api_error(logger, "order/placeOrder (close)", str(exc))
        if chat_id:
            notifier.send(f"Close failed for {pos.symbol}: {exc}", chat_id=chat_id)


def open_position(
    api: WeexAPI,
    logger,
    notifier: TelegramNotifier,
    symbol: str,
    side: str,
    last_price: float,
    chat_id: Optional[str],
    size_override: Optional[str] = None,
    reason: str = "",
) -> bool:
    side = side.lower()
    type_code = TYPE_OPEN_LONG if side == "buy" else TYPE_OPEN_SHORT
    size_info = (size_override, float(size_override)) if size_override else calc_size_for_order(symbol, last_price)
    if not size_info:
        logger.warning("Cannot calculate size for %s", symbol)
        return False
    size_str, size_float = size_info
    params = dict(
        symbol=symbol,
        client_oid=f"open-{uuid.uuid4().hex[:10]}",
        size=size_str,
        side=side,
        type_code=type_code,
        price=None,
        order_type="1",
        match_price="1",
    )

    logger.info("Opening %s %s at %.6f (size=%s). Reason: %s", symbol, side.upper(), last_price, size_str, reason)
    if settings.dry_run:
        return True

    try:
        api.set_leverage(symbol, settings.max_leverage, settings.max_leverage)
    except WeexAPIError as exc:
        logger.warning("Set leverage failed for %s: %s", symbol, exc)

    try:
        res = api.place_order(**params)
        logger.info("Open response: %s", res)
        if chat_id:
            notifier.send(
                f"Opened {symbol} {side.upper()} size={size_float:.6f} @ {last_price:.6f}\nReason: {reason}",
                chat_id=chat_id,
            )
        return True
    except WeexAPIError as exc:
        log_api_error(logger, "order/placeOrder (open)", str(exc))
        if chat_id:
            notifier.send(f"Open failed for {symbol}: {exc}", chat_id=chat_id)
        return False


def process_telegram_commands(
    api: WeexAPI,
    notifier: TelegramNotifier,
    logger,
    symbols: List[str],
    symbol_states: Dict[str, PositionState],
    last_update_id: Optional[int],
) -> Tuple[Optional[int], bool]:
    updates = notifier.get_updates(offset=last_update_id, timeout=0)
    stop_requested = False
    if not updates:
        return last_update_id, stop_requested

    positions_cache: Optional[List[PositionInfo]] = None

    for upd in updates:
        last_update_id = upd["update_id"] + 1
        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            continue
        text = (msg.get("text") or "").strip()
        chat_id = msg.get("chat", {}).get("id")
        if not text:
            continue
        lower = text.lower()

        if lower.startswith("/help"):
            notifier.send(format_help(), chat_id=chat_id)
            continue
        if lower.startswith("/stop"):
            stop_requested = True
            notifier.send("Stop requested. Finishing loop...", chat_id=chat_id)
            continue
        if lower.startswith("/status"):
            positions_cache = positions_cache or read_all_positions(api, logger)
            notifier.send(format_status_text(api, symbols, symbol_states, positions_cache), chat_id=chat_id)
            continue
        if lower.startswith("/balance"):
            positions_cache = positions_cache or read_all_positions(api, logger)
            notifier.send(format_balance_report(api, positions_cache), chat_id=chat_id)
            continue
        if lower.startswith("/close all"):
            positions_cache = positions_cache or read_all_positions(api, logger)
            for p in positions_cache:
                try:
                    last_price = get_last_price_from_ticker(api.get_ticker(p.symbol)) or 0.0
                    close_position_market(api, logger, notifier, p, last_price, str(chat_id), "manual /close all")
                except Exception as exc:
                    notifier.send(f"Close failed for {p.symbol}: {exc}", chat_id=chat_id)
            continue
        if lower.startswith("/close "):
            parts = lower.split()
            if len(parts) == 2:
                coin = parts[1].replace("/", "").replace("\\", "")
                sym = f"cmt_{coin}usdt" if not coin.startswith("cmt_") else coin
                positions_cache = positions_cache or read_all_positions(api, logger)
                for p in positions_cache:
                    if p.symbol.lower() == sym:
                        try:
                            last_price = get_last_price_from_ticker(api.get_ticker(p.symbol)) or 0.0
                            close_position_market(api, logger, notifier, p, last_price, str(chat_id), "manual close")
                        except Exception as exc:
                            notifier.send(f"Close failed for {p.symbol}: {exc}", chat_id=chat_id)
                        break
    return last_update_id, stop_requested


def evaluate_open_positions(
    api: WeexAPI,
    notifier: TelegramNotifier,
    logger,
    positions: List[PositionInfo],
    symbol_states: Dict[str, PositionState],
    chat_id_default: Optional[str],
) -> int:
    now = time.time()
    closed_count = 0

    for pos in positions:
        state = symbol_states.setdefault(pos.symbol, PositionState())
        if state.open_ts is None:
            state.open_ts = now
        try:
            price = get_last_price_from_ticker(api.get_ticker(pos.symbol))
        except Exception:
            price = None
        if price is None or price <= 0:
            continue

        roe_pct = calc_roe_pct(pos, price)
        minutes_open = (now - state.open_ts) / 60.0

        # trailing update
        if roe_pct >= settings.trailing_start_pct * 100:
            candidate = roe_pct - settings.trailing_step_pct * 100
            if state.trailing_stop_pct is None or candidate > state.trailing_stop_pct:
                state.trailing_stop_pct = candidate

        # stagnation tracker
        if state.stagnation_ref_price is None:
            state.stagnation_ref_price = price
            state.last_move_ts = now
        else:
            move_pct = abs(price - state.stagnation_ref_price) / state.stagnation_ref_price
            if move_pct >= settings.stagnation_move_pct:
                state.stagnation_ref_price = price
                state.last_move_ts = now
        stagnant = (
            state.last_move_ts is not None
            and (now - state.last_move_ts) / 60.0 >= settings.stagnation_minutes
        )

        close_reason = None
        if roe_pct <= -settings.stop_loss_pct * 100:
            close_reason = f"stop loss {roe_pct:+.2f}%"
        elif roe_pct >= settings.take_profit_pct * 100:
            close_reason = f"take profit {roe_pct:+.2f}%"
        elif state.trailing_stop_pct is not None and roe_pct <= state.trailing_stop_pct:
            close_reason = f"trailing stop {roe_pct:+.2f}% <= {state.trailing_stop_pct:.2f}%"
        elif stagnant and roe_pct >= settings.min_close_profit_pct * 100:
            close_reason = f"stagnation close {roe_pct:+.2f}%"

        if close_reason:
            close_position_market(api, logger, notifier, pos, price, chat_id_default, close_reason)
            closed_count += 1
            continue

        # AI exit check for mid-range PnL or long duration
        if roe_pct >= settings.min_close_profit_pct * 100 or minutes_open >= settings.stagnation_minutes:
            ai_exit = get_exit_decision(pos.symbol, pos.side, pos.entry_price or 0.0, price, roe_pct, minutes_open)
            if ai_exit == "CLOSE":
                close_position_market(api, logger, notifier, pos, price, chat_id_default, "AI exit")
                closed_count += 1
            elif ai_exit == "FLIP":
                # Close then reopen opposite if allowed in entry loop.
                close_position_market(api, logger, notifier, pos, price, chat_id_default, "AI flip close")
                closed_count += 1
                symbol_states[pos.symbol].last_signal = AISignal(
                    side="sell" if pos.side.startswith("LONG") else "buy",
                    confidence=0.7,
                    reason="AI flip",
                )

    return closed_count


def evaluate_new_entries(
    api: WeexAPI,
    notifier: TelegramNotifier,
    logger,
    symbols: List[str],
    positions_open: int,
    symbol_states: Dict[str, PositionState],
    chat_id_default: Optional[str],
):
    open_slots = settings.max_open_positions - positions_open
    if open_slots <= 0:
        return

    signals: Dict[str, AISignal] = {}
    for sym in symbols:
        try:
            candles = api.get_candles(sym, settings.candle_granularity, limit=200)
            sig = get_ai_signal(sym, candles)
            symbol_states.setdefault(sym, PositionState()).last_signal = sig
            signals[sym] = sig
            log_reasoning(
                logger,
                f"Signal for {sym}",
                f"{sig.reason}, side={sig.side}, conf={sig.confidence:.3f}, tp={sig.suggested_tp_pct}, sl={sig.suggested_sl_pct}",
            )
        except Exception as exc:
            logger.warning("Signal failed for %s: %s", sym, exc)

    sorted_syms = sorted(
        [s for s in symbols if s in signals],
        key=lambda s: getattr(signals[s], "confidence", 0.0),
        reverse=True,
    )

    for sym in sorted_syms:
        if open_slots <= 0:
            break
        sig = signals[sym]
        if not sig.side or sig.confidence < settings.signal_threshold:
            continue

        try:
            ticker = api.get_ticker(sym)
            last_price = get_last_price_from_ticker(ticker)
        except WeexAPIError as exc:
            logger.warning("Ticker failed for %s: %s", sym, exc)
            continue
        if last_price is None or last_price <= 0:
            logger.warning("No valid price for %s", sym)
            continue

        if open_position(api, logger, notifier, sym, sig.side, last_price, chat_id_default, reason=sig.reason):
            st = symbol_states.setdefault(sym, PositionState())
            st.open_ts = time.time()
            st.trailing_stop_pct = None
            st.stagnation_ref_price = last_price
            st.last_move_ts = time.time()
            open_slots -= 1


def parse_args():
    parser = argparse.ArgumentParser(description="WEEX DeepSeek trading bot")
    parser.add_argument("--mode", choices=["run-bot", "test-api", "test-signals"], default="run-bot")
    return parser.parse_args()


def test_api(api: WeexAPI, logger):
    logger.info("Testing API connectivity...")
    for fn, name in (
        (api.get_server_time, "server_time"),
        (api.get_assets, "assets"),
        (api.get_positions, "positions"),
    ):
        try:
            data = fn()
            logger.info("%s: %s", name, data)
        except WeexAPIError as exc:
            logger.error("%s error: %s", name, exc)


def test_signals(api: WeexAPI, logger):
    for sym in settings.symbols:
        candles = api.get_candles(sym, settings.candle_granularity, limit=200)
        sig = get_ai_signal(sym, candles)
        log_reasoning(logger, f"Signal for {sym}", f"{sig.reason}, side={sig.side}, conf={sig.confidence:.3f}")


def run_bot_loop(api: WeexAPI, notifier: TelegramNotifier, logger, stats: BotStats):
    symbols = settings.symbols
    symbol_states: Dict[str, PositionState] = {sym: PositionState() for sym in symbols}
    chat_id_default = settings.telegram_chat_id or None
    last_update_id: Optional[int] = None

    logger.info(
        "Config: symbols=%s max_positions=%d leverage=%dx TP=%.1f%% SL=%.1f%% trailing_start=%.1f%% trailing_step=%.1f%% interval=%ds",
        ",".join(symbols),
        settings.max_open_positions,
        settings.max_leverage,
        settings.take_profit_pct * 100,
        settings.stop_loss_pct * 100,
        settings.trailing_start_pct * 100,
        settings.trailing_step_pct * 100,
        settings.poll_interval_seconds,
    )

    if chat_id_default:
        notifier.send("[OK] Bot started.", chat_id=chat_id_default)

    while True:
        try:
            last_update_id, stop_requested = process_telegram_commands(
                api, notifier, logger, symbols, symbol_states, last_update_id
            )
            if stop_requested:
                logger.info("Stop requested from Telegram.")
                if chat_id_default:
                    notifier.send("[STOP] Bot stopped by /stop", chat_id=chat_id_default)
                break

            positions = read_all_positions(api, logger)
            positions_open = len(positions)

            closed = evaluate_open_positions(api, notifier, logger, positions, symbol_states, chat_id_default)
            positions_open = max(0, positions_open - closed)

            evaluate_new_entries(api, notifier, logger, symbols, positions_open, symbol_states, chat_id_default)

            sleep_left = settings.poll_interval_seconds
            while sleep_left > 0:
                last_update_id, stop_requested = process_telegram_commands(
                    api, notifier, logger, symbols, symbol_states, last_update_id
                )
                if stop_requested:
                    logger.info("Stop requested during sleep.")
                    if chat_id_default:
                        notifier.send("[STOP] Bot stopped by /stop", chat_id=chat_id_default)
                    return
                t = min(5, sleep_left)
                time.sleep(t)
                sleep_left -= t

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (CTRL+C)")
            if chat_id_default:
                notifier.send("[STOP] Bot terminated manually.", chat_id=chat_id_default)
            break
        except WeexAPIError as exc:
            logger.exception("Fatal WeexAPI error: %s", exc)
            if chat_id_default:
                notifier.send(f"[ERROR] Bot crashed with WeexAPIError: {exc}", chat_id=chat_id_default)
            time.sleep(5)
        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            if chat_id_default:
                notifier.send(f"[ERROR] Bot crashed with unexpected error: {exc}", chat_id=chat_id_default)
            time.sleep(5)


def main():
    args = parse_args()
    logger = setup_logger(settings.log_level)
    api = FakeWeexAPI() if settings.fake_weex_mode else WeexAPI()
    notifier = TelegramNotifier(TelegramConfig.from_settings())
    stats = BotStats()

    if args.mode == "test-api":
        test_api(api, logger)
        return
    if args.mode == "test-signals":
        test_signals(api, logger)
        return

    if not settings.dry_run and (not settings.api_key or not settings.api_secret or not settings.api_passphrase):
        logger.error("API credentials are missing. Set WEEX_API_KEY / WEEX_API_SECRET / WEEX_PASSPHRASE")
        notifier.send("[ERROR] Bot stopped: missing WEEX API credentials", chat_id=settings.telegram_chat_id or None)
        return

    run_bot_loop(api, notifier, logger, stats)


if __name__ == "__main__":
    main()
