from dataclasses import dataclass, field
import time

@dataclass
class BotStats:
    start_time: float = field(default_factory=time.time)
    trades_opened: int = 0
    trades_closed: int = 0

    def register_trade_open(self):
        self.trades_opened += 1

    def register_trade_closed(self):
        self.trades_closed += 1
