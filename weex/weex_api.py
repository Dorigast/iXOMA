import time
import hmac
import hashlib
import base64
import json
from typing import Any, Dict, Optional, Union, List

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
            # У WEEX успех обычно code "0" или "200"
            if code not in {"0", "200"} and resp.status_code != 200:
                raise WeexAPIError(f"API error {resp.status_code}: {data}")

        return data

    # ---- Public & account methods ----

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

    def get_contract_info(self, symbol: Optional[str] = None):
        query: Dict[str, Any] = {}
        if symbol:
            query["symbol"] = symbol
        return self._request("GET", "/capi/v2/market/contracts", query=query or None)

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
        price: Optional[str] = None,
        order_type: str = "0",
        match_price: str = "0",
        open_long: bool = True,
    ):
        # open_long=True -> type=1 (open long), False -> type=3 (close long)
        type_flag = "1" if open_long else "3"
        body: Dict[str, Any] = {
            "symbol": symbol,
            "client_oid": client_oid,
            "size": size,
            "type": type_flag,
            "order_type": order_type,
            "match_price": match_price,
        }
        if price is not None:
            body["price"] = price
        return self._request("POST", "/capi/v2/order/placeOrder", body=body, auth=True)
