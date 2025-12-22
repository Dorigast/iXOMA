import numpy as np
import pandas as pd

def generate_synthetic_prices(n=200):
    prices = [100]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.003)))
    return prices

def ema(series, span):
    return series.ewm(span=span).mean()

def backtest():
    prices = generate_synthetic_prices()
    df = pd.DataFrame({"close": prices})
    df["ema_fast"] = ema(df["close"], 10)
    df["ema_slow"] = ema(df["close"], 30)
    df["signal"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["ret"] = df["close"].pct_change().fillna(0)
    df["strategy_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]
    cum = (1 + df["strategy_ret"]).cumprod() - 1
    print("Backtest cumulative return:", round(cum.iloc[-1] * 100, 2), "%")

if __name__ == "__main__":
    backtest()
