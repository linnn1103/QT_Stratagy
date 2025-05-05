import logging
from typing import Optional
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataFetcher:
    BASE_URL = "https://api4.binance.com/api/v3/klines"

    def __init__(
        self,
        symbol: str = "SOLUSDT",
        interval: str = "15m",
        limit: int = 1000,
    ):
        self.symbol = symbol
        self.interval = interval
        self.limit = limit

        # Session with retry
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)

    def fetch_klines(
        self,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical K-line data from Binance.

        Args:
            start_time (int, optional): 起始時間戳(ms)
            end_time (int, optional): 結束時間戳(ms)

        Returns:
            pd.DataFrame: timestamp, open, high, low, close, volume, quote_volume, …
        """
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": self.limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        logger.info("Fetching %s klines for %s@%s", self.limit, self.symbol, self.interval)
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Error fetching klines: %s", e)
            raise

        # 格式化並轉 DataFrame
        df = pd.DataFrame([
            {
                "timestamp": pd.to_datetime(candle[0], unit="ms"),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "close_time": pd.to_datetime(candle[6], unit="ms"),
                "quote_volume": float(candle[7]),
            }
            for candle in data
        ])
        return df
