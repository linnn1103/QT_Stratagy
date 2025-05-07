from data.fetch_data import DataFetcher
from strategy.structure import SMCStrategy
from utils.indicator import Indicator
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json
import threading
import pandas as pd
import websocket
import collections

class WebSocketDataFetcher:
    """
        symbol (str): Trading pair symbol, e.g. "BTCUSDT".
        interval (str): Kline interval, e.g. "15m".
        on_candle (callable): Callback function that receives a dict
            with keys: timestamp, open, high, low, close, volume, close_time, quote_volume.
    """
    STREAM_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "SOLUSDT", interval: str = "15m", on_candle=None):
        self.symbol = symbol
        self.interval = interval
        self.on_candle = on_candle or (lambda x: print(x))
        self._ws = None

    def _build_stream_endpoint(self) -> str:
        # Binance requires lowercase symbol in stream path
        return f"{self.STREAM_URL}/{self.symbol.lower()}@kline_{self.interval}"

    def _on_message(self, ws, message):
        data = json.loads(message)
        k = data.get("k", {})
        if k.get("x", False):  # x = is this kline final?
            candle = {
                "timestamp": pd.to_datetime(k["t"], unit='ms'),
                "open": float(k["o"]),
                "high": float(k["h"]),
                "low": float(k["l"]),
                "close": float(k["c"]),
                "volume": float(k["v"]),
                "close_time": pd.to_datetime(k["T"], unit='ms'),
                "quote_volume": float(k["q"]),
            }
            self.on_candle(candle)

    def _on_error(self, ws, error):
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print(f"WebSocket closed: {close_status_code} - {close_msg}")

    def _on_open(self, ws):
        print(f"Connected to {self._build_stream_endpoint()}")

    def start(self):
        """
        Starts the WebSocket connection in a background thread.
        """
        endpoint = self._build_stream_endpoint()
        self._ws = websocket.WebSocketApp(
            endpoint,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
            on_open=self._on_open
        )
        thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """
        Closes the WebSocket connection.
        """
        if self._ws:
            self._ws.close()

def Initialize(symbol='SOLUSDT', interval='15m', limit=1000):
    """
    Initializes the DataFetcher with the given symbol, interval, and callback function.
    """
    old_data = DataFetcher(symbol=symbol, interval=interval, limit=limit).fetch_klines()
    return old_data

def get_smc_df(df, atr_period, swing_window, fvg_window, vol_mul, slope_lookback):
    smc = SMCStrategy(
        atr_period=atr_period,
        swing_window=swing_window,
        fvg_window=fvg_window,
        volatility_multiplier=vol_mul
    )
    df = smc.run(df)
    df['ADX'] = Indicator(df).get_ADX()
    df['ATR'] = Indicator(df).get_ATR()
    df['ADX_slope'] = np.nan
    for i in range(slope_lookback - 1, len(df)):
        y = df['ADX'].iloc[i - slope_lookback + 1: i + 1].values
        x = np.arange(slope_lookback)
        df.at[df.index[i], 'ADX_slope'] = np.polyfit(x, y, 1)[0]

    df['timestamp'] = (
        df['timestamp']
            .dt.tz_localize('UTC')
            .dt.tz_convert('Asia/Taipei')
            .dt.tz_localize(None)
    )

    # BOS detection
    df['BOS'] = False
    last_swing_high = None
    last_swing_low = None
    for idx, row in df.iterrows():
        if row['swing_high']:
            last_swing_high = row['high']
        if row['swing_low']:
            last_swing_low = row['low']
        if last_swing_high is not None and row['close'] > last_swing_high:
            df.at[idx, 'BOS'] = True
            last_swing_high = None
        if last_swing_low is not None and row['close'] < last_swing_low:
            df.at[idx, 'BOS'] = True
            last_swing_low = None

    # CHOCH detection
    df['CHOCH'] = False
    prev_bos_dir = None
    for idx, row in df.iterrows():
        if row['BOS']:
            curr_dir = 'bullish' if row['close'] > row['open'] else 'bearish'
            if prev_bos_dir and curr_dir != prev_bos_dir:
                df.at[idx, 'CHOCH'] = True
            prev_bos_dir = curr_dir
    df.loc[df['CHOCH'], 'BOS'] = False
    return df

if __name__ == "__main__":
    closes = Initialize()
    df = get_smc_df(
        df=closes,
        atr_period=14,
        swing_window=5,
        fvg_window=5,
        vol_mul=1.5,
        slope_lookback=14
    )
    closes = collections.deque(closes, maxlen=5000)
    print("舊資料處理完畢")
    def handle_candle(candle):
        closes.append(candle['close'])
        print(f"{candle['timestamp']}: Close = {candle['close']}")

    fetcher = WebSocketDataFetcher(symbol="SOLUSDT", interval="15m", on_candle=handle_candle)
    fetcher.start()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        fetcher.stop()