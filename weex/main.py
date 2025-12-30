import argparse
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Optional, List, Dict, Tuple

from config import settings
from utils.logger import setup_logger, log_reasoning, log_api_error
from utils.telegram_notifier import TelegramConfig, TelegramNotifier
from utils.stats import BotStats
from weex_api import WeexAPI, WeexAPIError
from strategy.simple_signal import moving_average_signal


@dataclass
class PositionInfo:
    symbol: str
    side: str
    size: float
    entry_price: Optional[float]
    margin: float
    unrealized_pnl: float


@dataclass
class SymbolState:
    open_ts: Optional[float] = None
    last_signal_side: Optional[str] = None
    last_signal_conf: float = 0.0
    last_signal_reason: str = ""
    tp_pct: float = 0.02
    sl_pct: float = 0.05


IDLE_FORCE_MINUTES = 20
AUTO_CLOSE_MINUTES = 60


def parse_positions_response(raw, logger=None) -> List[PositionInfo]:
    items = []

    if isinstance(raw, dict):
        data = raw.get("data") or raw
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            inner = data.get("positions") or data.get("list") or []
            if isinstance(inner, list):
                items = inner
    elif isinstance(raw, list):
        items = raw

    result: List[PositionInfo] = []
    for p in items or []:
        if not isinstance(p, dict):
            continue

        symbol = p.get("symbol")
        if not symbol:
            continue

        size_str = p.get("size") or p.get("positionSize") or p.get("availQty") or "0"
        try:
            size = float(size_str)
        except Exception:
            size = 0.0
        
        if size <= 0:
            continue

        side = p.get("side") or "LONG"

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
    except Exception as e:
        if logger:
            logger.error(f"Error reading positions: {e}")
        return []
    return parse_positions_response(raw, logger)


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


def estimate_volatility_pct(candles: List) -> float:
    if not candles:
        return 0.0
    ranges = []
    for c in candles:
        try:
            if isinstance(c, dict):
                high = float(c.get("high"))
                low = float(c.get("low"))
                close = float(c.get("close"))
            else:
                high = float(c[2])
                low = float(c[3])
                close = float(c[4])
            if close <= 0:
                continue
            r = (high - low) / close
            if r < 0:
                continue
            ranges.append(r)
        except Exception:
            continue
    if not ranges:
        return 0.0
    return sum(ranges) / len(ranges)


def compute_smart_tp_sl(base_tp: float, base_sl: float, vol: float) -> Tuple[float, float]:
    factor = 1.0 + max(min(vol * 50, 0.5), -0.5)
    tp = base_tp * factor
    sl = base_sl * factor
    return tp, sl


def get_fixed_order_size(symbol: str) -> Tuple[str, float]:
    size_str = settings.fixed_order_sizes.get(symbol)
    if not size_str:
        for key, val in settings.fixed_order_sizes.items():
            if key in symbol or symbol in key:
                size_str = val
                break
    if not size_str:
        size_str = "0.0001" 
    return size_str, float(size_str)


def close_position_market(api: WeexAPI, logger, symbol: str, size: float, last_price: float, reason: str):
    if size <= 0:
        return
    client_oid = f"close-{uuid.uuid4().hex[:10]}"
    params = dict(
        symbol=symbol,
        client_oid=client_oid,
        size=str(size),
        side="sell",
        price=str(last_price),
        order_type="0",
        match_price="0",
        open_long=False,
    )
    logger.info("Placing CLOSE order (%s): %s", reason, params)
    res = api.place_order(**params)
    logger.info("Close order response: %s", res)


def parse_args():
    p = argparse.ArgumentParser(description="WEEX AI Wars trading bot")
    p.add_argument(
        "--mode",
        choices=["run-bot", "test-api", "test-signals"],
        default="run-bot",
    )
    return p.parse_args()


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
        sig = moving_average_signal(
            candles,
            short_window=settings.short_window,
            long_window=settings.long_window,
            threshold=settings.signal_threshold,
            use_ema_filter=settings.use_ema_filter,
            use_ma_filter=settings.use_ma_filter,
            use_volume_filter=settings.use_volume_filter,
        )
        vol = estimate_volatility_pct(candles)
        log_reasoning(
            logger,
            f"Signal for {sym}",
            f"{sig.reason}, confidence={sig.confidence:.3f}, regime={sig.regime}, vol≈{vol*100:.2f}%",
        )


def format_help() -> str:
    lines = [
        "/help - помощь",
        "/status - текущие позиции и TP/SL",
        "/balance - баланс, маржа и PnL",
        "/close - меню закрытия позиций",
        "/close all - закрыть всё",
        "/stop - остановить бота",
    ]
    return "\n".join(lines)


# --- ОБНОВЛЕННАЯ ЛОГИКА СТАТУСА ---
def format_status_text(
    api: WeexAPI,
    symbols: List[str],
    symbol_states: Dict[str, SymbolState],
    balance: Optional[float],
    equity: Optional[float],
    positions: List[PositionInfo],
) -> str:
    lines: List[str] = []
    lines.append("📈Market Status:")
    pos_by_symbol = {p.symbol: p for p in positions}

    for sym in symbols:
        pos = pos_by_symbol.get(sym)
        # Получаем состояния для TP/SL
        st = symbol_states.get(sym) or SymbolState()
        vol = 0.0
        # Для скорости в статусе можно не запрашивать волатильность заново,
        # а использовать дефолтные TP/SL, либо кэшированные. 
        # Но если нужно точно, придется делать запрос (может замедлить ответ).
        # Используем быстрый расчет на основе конфига:
        tp_pct = settings.take_profit_pct
        sl_pct = settings.stop_loss_pct

        if pos is None:
             lines.append(f"⚪ {sym}: no open position")
        else:
            try:
                # Пытаемся получить текущую цену для красивого отчета
                last_price = get_last_price_from_ticker(api.get_ticker(sym))
            except Exception:
                last_price = None

            last = last_price or pos.entry_price or 0.0
            pnl_pct = 0.0
            if pos.entry_price and last:
                direction = 1.0 if (pos.side or "").upper().startswith("LONG") else -1.0
                pnl_pct = (last - pos.entry_price) / pos.entry_price * 100.0 * direction
            
            entry_str = f"{pos.entry_price:.4f}" if pos.entry_price else "0.0000"
            last_str = f"{last:.4f}" if last else "0.0000"
            
            # --- РАСЧЕТ ЦЕН TP и SL ---
            # Учитываем волатильность если данные были обновлены в цикле
            if st.tp_pct > 0:
                tp_pct = st.tp_pct
                sl_pct = st.sl_pct

            if pos.entry_price:
                take_profit_price = pos.entry_price * (1 + tp_pct)
                stop_loss_price = pos.entry_price * (1 - sl_pct)
            else:
                take_profit_price = 0.0
                stop_loss_price = 0.0
            
            lines.append(f"🟢 {sym} (LONG)")
            lines.append(f"   Size: {pos.size:.4f} | Margin: {pos.margin:.2f}$")
            lines.append(f"   Entry: `{entry_str}` ➔ Last: `{last_str}`")
            lines.append(f"   💰 PnL: {pos.unrealized_pnl:+.2f}$ ({pnl_pct:+.2f}%)")
            lines.append(f"   🎯 TP: `{take_profit_price:.4f}` (+{tp_pct*100:.1f}%)")
            lines.append(f"   🛑 SL: `{stop_loss_price:.4f}` (-{sl_pct*100:.1f}%)")
            lines.append("")

    return "\n".join(lines)


# --- ИСПРАВЛЕННАЯ ФУНКЦИЯ BALANCE ---
def format_balance_report(api: WeexAPI, positions: List[PositionInfo]) -> str:
    try:
        assets = api.get_assets()
        # API может вернуть словарь {"data": [...]} или сразу список [...]
        data = None
        if isinstance(assets, dict):
            data = assets.get("data") or assets
        elif isinstance(assets, list):
            data = assets
        
        row = None
        if isinstance(data, list) and len(data) > 0:
            row = data[0]
        elif isinstance(data, dict):
            row = data
        
        if not row:
             return "⚠️ Balance data is empty or format changed."

        balance = float(row.get("balance") or row.get("total_balance") or 0.0)
        equity = float(row.get("equity") or row.get("totalEquity") or row.get("total_equity") or 0.0)
        available = float(row.get("available") or row.get("availableBalance") or row.get("available_balance") or 0.0)

    except Exception as e:
        return f"⚠️ Error fetching balance info: {e}"

    total_margin = sum(p.margin for p in positions)
    total_unrealized_pnl = sum(p.unrealized_pnl for p in positions)
    
    lines = ["🏦 Wallet Balance"]
    lines.append(f"💵 Equity (Total): {equity:.2f} USDT")
    lines.append(f"🔓 Available: {available:.2f} USDT")
    lines.append("")
    lines.append(f"🔒 In Trades (Margin): {total_margin:.2f} USDT")
    
    if total_unrealized_pnl >= 0:
        pnl_icon = "🟢"
    else:
        pnl_icon = "🔴"
        
    lines.append(f"{pnl_icon} **Unrealized PnL:** {total_unrealized_pnl:+.2f} USDT")
    
    return "\n".join(lines)


def run_bot_loop(api: WeexAPI, notifier: TelegramNotifier, logger, stats: BotStats):
    allowed = {"cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", "cmt_bnbusdt", "cmt_xrpusdt"}
    symbols = [s for s in settings.symbols if s in allowed] or ["cmt_btcusdt"]

    logger.info("Starting trading loop on symbols: %s", ", ".join(symbols))
    
    logger.info(
        "Config: short_MA=%d, long_MA=%d, FIXED_SIZES=%s, dry_run=%s, base_TP=%.2f%%, base_SL=%.2f%%, "
        "max_open_positions=%d, granularity=%s, poll_interval=%d, trailing_start=%.2f%%, trailing_step=%.2f%%, "
        "signal_threshold=%.3f, force_entry=%s, leverage=%dx",
        settings.short_window,
        settings.long_window,
        settings.fixed_order_sizes,
        settings.dry_run,
        settings.take_profit_pct * 100,
        settings.stop_loss_pct * 100,
        settings.max_open_positions,
        settings.candle_granularity,
        settings.poll_interval_seconds,
        settings.trailing_start_pct * 100,
        settings.trailing_step_pct * 100,
        settings.signal_threshold,
        settings.force_entry,
        settings.max_leverage,
    )

    symbol_states: Dict[str, SymbolState] = {sym: SymbolState() for sym in symbols}
    tg_last_update_id: Optional[int] = None
    pending_close: Dict[int, Dict[str, object]] = {}

    chat_id_default = settings.telegram_chat_id or None
    if chat_id_default:
        notifier.send("🟢 Bot started.", chat_id=chat_id_default)

    def process_telegram_commands() -> Tuple[bool, bool]:
        nonlocal tg_last_update_id, pending_close
        stop_requested = False
        manual_closed_any = False

        updates = notifier.get_updates(offset=tg_last_update_id, timeout=0)
        if not updates:
            return stop_requested, manual_closed_any

        for upd in updates:
            tg_last_update_id = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            
            # ИГНОР СТАРЫХ СООБЩЕНИЙ
            msg_date = msg.get("date")
            if msg_date and msg_date < stats.start_time:
                continue

            chat = msg.get("chat") or {}
            chat_id = chat.get("id")
            text = (msg.get("text") or "").strip()
            if not text:
                continue
            lower = text.lower()

            # ВАЖНО: Мы НЕ вызываем read_all_positions здесь для каждого сообщения.
            # Это замедляло бота. Мы вызываем его только внутри нужных команд.

            if lower in ("yes", "да") and chat_id in pending_close:
                # Тут нужно прочитать позиции, чтобы закрыть
                positions = read_all_positions(api, logger)
                req = pending_close.pop(chat_id)
                symbols_to_close: List[str] = req.get("symbols", [])  # type: ignore
                for p in positions:
                    if not req.get("all") and p.symbol not in symbols_to_close:
                        continue
                    try:
                        ticker = api.get_ticker(p.symbol)
                        last_price = get_last_price_from_ticker(ticker)
                        if last_price is None or last_price <= 0:
                            continue
                        close_position_market(api, logger, p.symbol, p.size, last_price, "TG CONFIRMED CLOSE")
                        manual_closed_any = True
                        notifier.send(
                            f"⏹ Closed {p.symbol} size={p.size:.6f} at {last_price}",
                            chat_id=chat_id,
                        )
                    except Exception as exc:
                        notifier.send(f"⚠️ Error closing {p.symbol}: {exc}", chat_id=chat_id)
                continue

            if lower in ("no", "нет") and chat_id in pending_close:
                pending_close.pop(chat_id, None)
                notifier.send("❎ Close cancelled", chat_id=chat_id)
                continue

            if lower.startswith("/help"):
                notifier.send(format_help(), chat_id=chat_id)
                continue

            if lower.startswith("/stop"):
                notifier.send("⏹ Stop requested, finishing loop...", chat_id=chat_id)
                stop_requested = True
                continue

            if lower.startswith("/status"):
                # Запрашиваем позиции только по требованию
                positions = read_all_positions(api, logger)
                text_status = format_status_text(api, symbols, symbol_states, None, None, positions)
                notifier.send(text_status, chat_id=chat_id)
                continue

            if lower.startswith("/balance"):
                # Запрашиваем позиции и баланс только по требованию
                positions = read_all_positions(api, logger)
                text_bal = format_balance_report(api, positions)
                notifier.send(text_bal, chat_id=chat_id)
                continue

            if lower.strip() == "/close":
                positions = read_all_positions(api, logger)
                if not positions:
                    notifier.send("No open positions to close.", chat_id=chat_id)
                    continue
                lines = ["Open positions:"]
                for p in positions:
                    short = p.symbol.replace("cmt_", "").replace("usdt", "")
                    lines.append(f"{p.symbol} (/{'close ' + short}) size={p.size:.6f}")
                lines.append("")
                lines.append("Use /close btc (eth/sol/bnb/xrp) or /close all, then reply YES to confirm.")
                notifier.send("\n".join(lines), chat_id=chat_id)
                continue

            if lower.strip() == "/close all":
                positions = read_all_positions(api, logger)
                if not positions:
                    notifier.send("No open positions to close.", chat_id=chat_id)
                    continue
                pending_close[chat_id] = {"all": True, "symbols": [], "ts": time.time()}
                notifier.send("⚠️ Confirm closing ALL positions: reply YES (or NO).", chat_id=chat_id)
                continue

            if lower.startswith("/close "):
                positions = read_all_positions(api, logger)
                parts = lower.split()
                if len(parts) == 2:
                    coin = parts[1]
                    syms = [
                        p.symbol
                        for p in positions
                        if p.symbol.replace("cmt_", "").replace("usdt", "").lower() == coin
                    ]
                    if not syms:
                        notifier.send(f"No open positions for {coin.upper()}.", chat_id=chat_id)
                        continue
                    pending_close[chat_id] = {"all": False, "symbols": syms, "ts": time.time()}
                    notifier.send(
                        f"⚠️ Confirm closing positions for {coin.upper()} ({', '.join(syms)}): reply YES (or NO).",
                        chat_id=chat_id,
                    )
                    continue

        return stop_requested, manual_closed_any

    while True:
        try:
            stop_req, _ = process_telegram_commands()
            if stop_req:
                logger.info("Stop requested from Telegram. Exiting loop.")
                if chat_id_default:
                    notifier.send("🔴 Bot stopped by /stop command.", chat_id=chat_id_default)
                break

            now = time.time()

            # Основной цикл обновляет позиции для проверки авто-закрытия
            positions = read_all_positions(api, logger)
            open_positions_count = len(positions)
            pos_by_symbol: Dict[str, PositionInfo] = {p.symbol: p for p in positions}
            
            for p in positions:
                st = symbol_states.get(p.symbol)
                if not st:
                    st = SymbolState()
                    symbol_states[p.symbol] = st
                if st.open_ts is None:
                    st.open_ts = now
                age_min = (now - st.open_ts) / 60.0
                if age_min >= AUTO_CLOSE_MINUTES:
                    try:
                        ticker = api.get_ticker(p.symbol)
                        last_price = get_last_price_from_ticker(ticker)
                        if last_price is None or last_price <= 0:
                            continue
                        close_position_market(api, logger, p.symbol, p.size, last_price, "AUTO TIME CLOSE 60m")
                        if chat_id_default:
                            notifier.send(
                                f"🕒 Auto-close {p.symbol} after {AUTO_CLOSE_MINUTES} minutes",
                                chat_id=chat_id_default,
                            )
                    except Exception as exc:
                        if chat_id_default:
                            notifier.send(f"⚠️ Auto-close {p.symbol} failed: {exc}", chat_id=chat_id_default)

            signals: Dict[str, object] = {}
            for sym in symbols:
                candles = api.get_candles(sym, settings.candle_granularity, limit=200)
                sig = moving_average_signal(
                    candles,
                    short_window=settings.short_window,
                    long_window=settings.long_window,
                    threshold=settings.signal_threshold,
                    use_ema_filter=settings.use_ema_filter,
                    use_ma_filter=settings.use_ma_filter,
                    use_volume_filter=settings.use_volume_filter,
                )
                vol = estimate_volatility_pct(candles)
                tp_pct, sl_pct = compute_smart_tp_sl(settings.take_profit_pct, settings.stop_loss_pct, vol)
                st = symbol_states[sym]
                st.last_signal_side = sig.side
                st.last_signal_conf = sig.confidence
                st.last_signal_reason = sig.reason
                st.tp_pct = tp_pct
                st.sl_pct = sl_pct

                log_reasoning(
                    logger,
                    f"Signal for {sym}",
                    f"{sig.reason}, confidence={sig.confidence:.3f}, regime={sig.regime}, vol≈{vol*100:.2f}%",
                )
                signals[sym] = sig

            positions_opened_this_cycle = 0
            sorted_symbols = sorted(symbols, key=lambda s: getattr(signals[s], "confidence", 0.0), reverse=True)

            for sym in sorted_symbols:
                if open_positions_count >= settings.max_open_positions:
                    break
                if positions_opened_this_cycle >= 1:
                    break
                if sym in pos_by_symbol:
                    continue

                sig = signals[sym]
                if not (sig.side == "buy" and sig.confidence >= settings.signal_threshold):
                    continue

                try:
                    ticker = api.get_ticker(sym)
                    last_price = get_last_price_from_ticker(ticker)
                except WeexAPIError as exc:
                    logger.warning("Error ticker for %s: %s", sym, exc)
                    continue
                if last_price is None or last_price <= 0:
                    logger.warning("No valid last price for %s, skip", sym)
                    continue

                try:
                    api.set_leverage(sym, settings.max_leverage, settings.max_leverage)
                except WeexAPIError as exc:
                    if chat_id_default:
                        notifier.send(f"⚠️ Cannot set leverage for {sym}: {exc}", chat_id=chat_id_default)

                size_str, size_float = get_fixed_order_size(sym)

                client_oid = f"bot-{uuid.uuid4().hex[:10]}"
                if settings.dry_run:
                    logger.info(
                        "[DRY RUN] Would OPEN LONG %s size=%s at %.4f (TP≈%.1f%% SL≈%.1f%%) reason=%s",
                        sym,
                        size_str,
                        last_price,
                        symbol_states[sym].tp_pct * 100,
                        symbol_states[sym].sl_pct * 100,
                        sig.reason,
                    )
                else:
                    params = dict(
                        symbol=sym,
                        client_oid=client_oid,
                        size=size_str,
                        side="buy",
                        price=str(last_price),
                        order_type="0",
                        match_price="0",
                        open_long=True,
                    )
                    logger.info("Placing order: %s", params)
                    try:
                        res = api.place_order(**params)
                        logger.info("Order response: %s", res)
                        if chat_id_default:
                            notifier.send(
                                f"🚀 OPEN LONG {sym}\nEntry={last_price:.4f}\nSize={size_float:.6f}",
                                chat_id=chat_id_default,
                            )
                        symbol_states[sym].open_ts = time.time()
                        open_positions_count += 1
                        positions_opened_this_cycle += 1
                        stats.register_trade_open()
                    except WeexAPIError as exc:
                        log_api_error(logger, "order/placeOrder (open long)", str(exc))
                        if chat_id_default:
                            notifier.send(f"⚠️ Error opening {sym}: {exc}", chat_id=chat_id_default)

            if settings.force_entry and open_positions_count == 0 and positions_opened_this_cycle == 0:
                sym = "cmt_btcusdt"
                try:
                    ticker = api.get_ticker(sym)
                    last_price = get_last_price_from_ticker(ticker)
                except WeexAPIError:
                    last_price = None

                if last_price is not None and last_price > 0:
                    size_str, size_float = get_fixed_order_size(sym)
                    
                    client_oid = f"force-{uuid.uuid4().hex[:10]}"

                    if not settings.dry_run:
                        try:
                            api.set_leverage(sym, settings.max_leverage, settings.max_leverage)
                        except WeexAPIError:
                            pass

                        params = dict(
                            symbol=sym,
                            client_oid=client_oid,
                            size=size_str,
                            side="buy",
                            price=str(last_price),
                            order_type="0",
                            match_price="0",
                            open_long=True,
                        )
                        logger.info("FORCED BTC ENTRY order: %s", params)
                        try:
                            res = api.place_order(**params)
                            logger.info("FORCED BTC response: %s", res)
                            if chat_id_default:
                                notifier.send(
                                    f"⚡ FORCED BTC ENTRY {sym}\nEntry={last_price:.4f}\nSize={size_float:.6f}",
                                    chat_id=chat_id_default,
                                )
                            symbol_states[sym].open_ts = time.time()
                        except WeexAPIError as exc:
                            log_api_error(logger, "order/placeOrder (forced btc)", str(exc))
                            if chat_id_default:
                                notifier.send(f"⚠️ Forced BTC entry failed: {exc}", chat_id=chat_id_default)

            sleep_left = settings.poll_interval_seconds
            while sleep_left > 0:
                stop_req, _ = process_telegram_commands()
                if stop_req:
                    logger.info("Stop requested from Telegram during sleep. Exiting loop.")
                    if chat_id_default:
                        notifier.send("🔴 Bot stopped by /stop command (during sleep).", chat_id=chat_id_default)
                    return
                t = min(3, sleep_left)
                time.sleep(t)
                sleep_left -= t

        except KeyboardInterrupt:
            logger.info("Bot stopped by user (CTRL+C)")
            if chat_id_default:
                notifier.send("⛔ Bot terminated manually.", chat_id=chat_id_default)
            break
        except WeexAPIError as exc:
            logger.exception("Fatal WeexAPI error: %s", exc)
            if chat_id_default:
                notifier.send(f"⚠️ Bot crashed with WeexAPIError: {exc}", chat_id=chat_id_default)
            time.sleep(5)
        except Exception as exc:
            logger.exception("Unexpected error in main loop: %s", exc)
            if chat_id_default:
                notifier.send(f"⚠️ Bot crashed with unexpected error: {exc}", chat_id=chat_id_default)
            time.sleep(5)


def main():
    args = parse_args()
    logger = setup_logger(settings.log_level)

    api = WeexAPI()
    tg_conf = TelegramConfig.from_settings()
    notifier = TelegramNotifier(tg_conf)

    stats = BotStats()

    if args.mode == "test-api":
        test_api(api, logger)
        return
    if args.mode == "test-signals":
        test_signals(api, logger)
        return

    if not settings.dry_run:
        if not settings.api_key or not settings.api_secret or not settings.api_passphrase:
            logger.error("API credentials are missing. Set WEEX_API_KEY / WEEX_API_SECRET / WEEX_PASSPHRASE")
            notifier.send("❌ Bot stopped: missing WEEX API credentials", chat_id=settings.telegram_chat_id or None)
            return

    run_bot_loop(api, notifier, logger, stats)


if __name__ == "__main__":
    main()