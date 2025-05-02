import requests
import pandas as pd

class DataFetcher:
    BASE_URL = "https://api4.binance.com/api/v3/klines"

    def __init__(self, symbol: str = "SOLUSDT", interval: str = "15m", limit: int = 1001):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

    def fetch_klines(self) -> list:
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit
        }
        resp = requests.get(self.BASE_URL, params=params)
        data = resp.json()

        formatted_data = []
        for candle in data:
            item = {
                "timestamp": pd.to_datetime(candle[0], unit='ms'),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "close_time": pd.to_datetime(candle[6], unit='ms'),
                "quote_volume": float(candle[7]),
            }
            formatted_data.append(item)
    
        return formatted_data