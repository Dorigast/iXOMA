import base64
import hashlib
import hmac
import json
import random
import time
from typing import Any, Dict, List, Optional, Union

import requests

from config import settings


class WeexAPIError(Exception):
    pass


class WeexAPI:
    def __init__(self):
        self.api_key = settings.api_key
        self.secret_key = settings.api_secret
        self.passphrase = settings.api_passphrase
        self.base_url = settings.base_url.rstrip("/")

    def _timestamp_ms(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, timestamp: str, method: str, path: str, query_string: str, body: str) -> str:
        message = timestamp + method.upper() + path
        if query_string:
            message += "?" + query_string
        message += body
        digest = hmac.new(self.secret_key.encode(), message.encode(), hashlib.sha256).digest()
        return base64.b64encode(digest).decode()

    def _headers(self, timestamp: str, signature: str) -> Dict[str, str]:
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    def _request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: bool = False,
    ) -> Union[Dict[str, Any], List[Any]]:
        query_string = ""
        if query:
            parts = []
            for k, v in query.items():
                if v is None:
                    continue
                parts.append(f"{k}={v}")
            query_string = "&".join(parts)

        body_str = ""
        if body:
            body_str = json.dumps(body, separators=(",", ":"), ensure_ascii=False)

        url = self.base_url + path
        if query_string:
            url += "?" + query_string

        timestamp = self._timestamp_ms()
        headers = {"Content-Type": "application/json", "locale": "en-US"}
        if auth:
            signature = self._sign(timestamp, method, path, query_string, body_str)
            headers = self._headers(timestamp, signature)

        try:
            resp = requests.request(method=method, url=url, headers=headers, data=body_str or None, timeout=10)
        except requests.exceptions.RequestException as exc:
            raise WeexAPIError(f"Network error: {exc!r}")

        try:
            data = resp.json()
        except Exception:
            raise WeexAPIError(f"Non-JSON response ({resp.status_code}): {resp.text}")

        if isinstance(data, dict):
            code = str(data.get("code", "0"))
            if code not in {"0", "200"} and resp.status_code != 200:
                raise WeexAPIError(f"API error {resp.status_code}: {data}")
        return data

    def get_server_time(self):
        return self._request("GET", "/capi/v2/market/time")

    def get_ticker(self, symbol: str):
        return self._request("GET", "/capi/v2/market/ticker", query={"symbol": symbol})

    def get_candles(self, symbol: str, granularity: str = "5m", limit: int = 100):
        return self._request(
            "GET",
            "/capi/v2/market/candles",
            query={"symbol": symbol, "granularity": granularity, "limit": limit},
        )

    def get_assets(self):
        return self._request("GET", "/capi/v2/account/assets", auth=True)

    def get_positions(self):
        return self._request("GET", "/capi/v2/account/position/allPosition", auth=True)

    def set_leverage(self, symbol: str, long_leverage: int, short_leverage: Optional[int] = None):
        if short_leverage is None:
            short_leverage = long_leverage
        body = {
            "symbol": symbol,
            "marginMode": 1,
            "longLeverage": str(long_leverage),
            "shortLeverage": str(short_leverage),
        }
        return self._request("POST", "/capi/v2/account/leverage", body=body, auth=True)

    def place_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        side: str,
        type_code: Union[str, int],  # 1=OpenLong, 2=OpenShort, 3=CloseLong, 4=CloseShort
        price: Optional[str] = None,
        order_type: str = "0",
        match_price: str = "0",
    ):
        body: Dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": size,
            "type": type_code,
            "side": side,
            "order_type": order_type,
            "match_price": match_price,
        }
        if price is not None:
            body["price"] = price
        return self._request("POST", "/capi/v2/order/placeOrder", body=body, auth=True)


class FakeWeexAPI:
    """
    Offline stub to let the bot run in dry-run mode without live API.
    """

    def __init__(self):
        self.base_prices = {sym: 100 + idx * 10 for idx, sym in enumerate(settings.symbols)}
        self.positions: Dict[str, Dict[str, Any]] = {}

    def _tick_price(self, symbol: str) -> float:
        base = self.base_prices.get(symbol, 100.0)
        change = random.uniform(-0.003, 0.003)
        base *= 1 + change
        self.base_prices[symbol] = base
        return base

    def get_server_time(self):
        now_ms = int(time.time() * 1000)
        return {
            "epoch": str(now_ms / 1000),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "timestamp": now_ms,
        }

    def get_ticker(self, symbol: str):
        last = self._tick_price(symbol)
        return {"data": {"last": last, "symbol": symbol}}

    def get_candles(self, symbol: str, granularity: str = "5m", limit: int = 100):
        now = int(time.time() // 60 * 60)
        price = self.base_prices.get(symbol, 100.0)
        candles = []
        for i in range(limit):
            t = now - (limit - i) * 300
            change = random.uniform(-0.003, 0.003)
            price *= 1 + change
            high = price * (1 + abs(random.uniform(0, 0.002)))
            low = price * (1 - abs(random.uniform(0, 0.002)))
            open_p = price * (1 - random.uniform(-0.001, 0.001))
            close = price
            vol = random.uniform(10, 100)
            amt = vol * close
            candles.append([t, open_p, high, low, close, vol, amt])
        self.base_prices[symbol] = price
        return candles

    def get_assets(self):
        return {"code": "0", "data": [{"balance": 1000, "equity": 1000, "available": 900}]}

    def get_positions(self):
        items = []
        for sym, p in self.positions.items():
            last = self.base_prices.get(sym, p["entry_price"])
            direction = 1.0 if p["side"] == "LONG" else -1.0
            unreal = (last - p["entry_price"]) * p["size"] * direction
            items.append(
                {
                    "symbol": sym,
                    "side": p["side"],
                    "size": p["size"],
                    "margin": p["margin"],
                    "unrealizedPnl": unreal,
                    "open_value": p["entry_price"] * p["size"],
                }
            )
        return {"code": "0", "data": {"positions": items}}

    def set_leverage(self, symbol: str, long_leverage: int, short_leverage: Optional[int] = None):
        return {"code": "0", "msg": "ok"}

    def place_order(
        self,
        symbol: str,
        client_oid: str,
        size: str,
        side: str,
        type_code: Union[str, int],
        price: Optional[str] = None,
        order_type: str = "1",
        match_price: str = "1",
    ):
        size_val = float(size)
        last = float(price) if price is not None else self.base_prices.get(symbol, 100.0)
        if str(type_code) in {"1", "2"}:
            side_label = "LONG" if side.lower() == "buy" else "SHORT"
            margin = size_val * last / max(settings.max_leverage, 1)
            self.positions[symbol] = {
                "side": side_label,
                "size": size_val,
                "entry_price": last,
                "margin": margin,
            }
        elif str(type_code) in {"3", "4"}:
            self.positions.pop(symbol, None)
        return {"code": "0", "msg": "ok", "client_oid": client_oid}
