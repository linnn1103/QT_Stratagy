import json
import threading
import pandas as pd
import websocket

class WebSocketDataFetcher:
    """
        symbol (str): Trading pair symbol, e.g. "BTCUSDT".
        interval (str): Kline interval, e.g. "15m".
        on_candle (callable): Callback function that receives a dict
            with keys: timestamp, open, high, low, close, volume, close_time, quote_volume.
    """
    STREAM_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "15m", on_candle=None):
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

if __name__ == "__main__":
    closes = []

    def handle_candle(candle):
        closes.append(candle['close'])
        print(f"{candle['timestamp']}: Close = {candle['close']}")

    fetcher = WebSocketDataFetcher(symbol="SOLUSDT", interval="15m", on_candle=handle_candle)
    fetcher.start()

    import time; time.sleep(600)
    fetcher.stop()
