import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BacktestDataSaver:
    _call_count = 0
    _csv_path = "solusdt_15m.csv"

    @classmethod
    def rr(cls):
        """
        每次呼叫回傳400筆資料，第一次為0~399，第二次為1~400，依此類推。
        """
        filename = cls._csv_path
        df = pd.read_csv(filename, parse_dates=["timestamp", "close_time"])
        float_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[float_cols] = df[float_cols].astype(float)
        logger.info("Loaded %d rows from %s", len(df), filename)
        start = cls._call_count
        end = cls._call_count + 400
        cls._call_count += 1
        return df.iloc[start:end].reset_index(drop=True)