from typing import Dict, Any
from functools import partial
from config import settings
from weex_api import WeexAPI

# Simple function-based tools compatible with CrewAI
def make_tools(api: WeexAPI):
    def get_price(symbol: str) -> Dict[str, Any]:
        return api.fetch_ticker(symbol)

    def get_order_book(symbol: str, limit: int = 50) -> Dict[str, Any]:
        return api.fetch_order_book(symbol, limit=limit)

    def place_order(symbol: str, side: str, amount: float, price: float | None = None) -> Dict[str, Any]:
        return api.create_order(symbol=symbol, side=side, order_type="limit" if price else "market", amount=amount, price=price)

    return {
        "get_price": get_price,
        "get_order_book": get_order_book,
        "place_order": place_order,
    }

def tool_descriptions():
    return {
        "get_price": "Fetch latest ticker for a symbol (expects symbol string like 'BTC/USDT').",
        "get_order_book": "Fetch order book snapshot for a symbol.",
        "place_order": "Place or simulate an order; respects DRY_RUN from config.",
    }
