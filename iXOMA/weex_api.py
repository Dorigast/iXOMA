import time
import ccxt
from typing import Any, Dict, List, Optional
from config import settings

class WeexAPI:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.exchange = ccxt.weex({
            "apiKey": settings.weex_api_key,
            "secret": settings.weex_api_secret,
            "password": settings.weex_password,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        })

    def _timestamp(self) -> int:
        return int(time.time() * 1000)

    def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        return self.exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, limit: int = 50) -> Dict[str, Any]:
        return self.exchange.fetch_order_book(symbol, limit=limit)

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 100) -> List[List[Any]]:
        return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "info": "dry-run",
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "amount": amount,
                "price": price,
                "params": params or {},
                "timestamp": self._timestamp(),
            }
        return self.exchange.create_order(symbol, order_type, side, amount, price, params or {})
