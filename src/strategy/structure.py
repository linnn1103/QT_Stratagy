from data.fetch_data import DataFetcher
import pandas as pd
import talib

# -- SMC Strategy Module --

class SMCStrategy:
    def __init__(
        self,
        symbol: str = "SOLUSDT",
        interval: str = "15m",
        limit: int = 1000,
        atr_period: int = 14,
        swing_window: int = 5,
        fvg_window: int = 3,
        volatility_multiplier: float = 1.0,
    ):
        # Initialize data fetcher and parameters
        self.fetcher = DataFetcher(symbol=symbol, interval=interval, limit=limit)
        self.atr_period = atr_period
        self.swing_window = swing_window
        self.fvg_window = fvg_window
        self.volatility_multiplier = volatility_multiplier

    def fetch_data(self) -> pd.DataFrame:
        """
        Fetches klines and calculates ATR.
        Returns a DataFrame with raw OHLCV and ATR.
        """
        df = self.fetcher.fetch_klines()
        df['atr'] = talib.ATR(
            df['high'], df['low'], df['close'], timeperiod=self.atr_period
        )
        return df

    def find_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify swing highs and lows using a rolling window.
        """
        df['swing_high'] = False
        df['swing_low'] = False
        w = self.swing_window
        for i in range(w, len(df) - w):
            window_highs = df['high'].iloc[i-w:i+w+1]
            window_lows = df['low'].iloc[i-w:i+w+1]
            if df['high'].iat[i] == window_highs.max():
                df.at[df.index[i], 'swing_high'] = True
            if df['low'].iat[i] == window_lows.min():
                df.at[df.index[i], 'swing_low'] = True
        return df

    def identify_structure_breaks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark structure break points when price breaks a prior swing high/low.
        """
        df['structure_break'] = None
        last_swing_high = None
        last_swing_low = None
        for idx, row in df.iterrows():
            if row['swing_high']:
                last_swing_high = row['high']
            if row['swing_low']:
                last_swing_low = row['low']

            if last_swing_high and row['close'] > last_swing_high:
                df.at[idx, 'structure_break'] = 'break_high'
            elif last_swing_low and row['close'] < last_swing_low:
                df.at[idx, 'structure_break'] = 'break_low'
        return df

    def detect_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Order Blocks (OB) based on high-volatility candles and close position.
        """
        df['bullish_ob'] = False
        df['bearish_ob'] = False
        # Use ATR * multiplier to filter high-vol candles
        threshold = df['atr'] * self.volatility_multiplier
        for i in range(1, len(df)):
            prev = df.iloc[i - 1]
            curr = df.iloc[i]
            # bullish OB: large range, close in upper half
            if (curr['high'] - curr['low']) > threshold.iat[i] and curr['close'] > (curr['low'] + curr['high'])/2:
                df.at[df.index[i], 'bullish_ob'] = True
            # bearish OB: large range, close in lower half
            if (curr['high'] - curr['low']) > threshold.iat[i] and curr['close'] < (curr['low'] + curr['high'])/2:
                df.at[df.index[i], 'bearish_ob'] = True
        return df

    def detect_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect Fair Value Gaps (FVG) by comparing three consecutive candles.
        """
        df['fvg_bullish'] = False
        df['fvg_bearish'] = False
        n = self.fvg_window
        for i in range(n, len(df)):
            c1, c2, c3 = df.iloc[i-n], df.iloc[i-n+1], df.iloc[i]
            # Bullish FVG: gap down then fill zone
            if c2['low'] > c1['high'] and c3['close'] > c2['low']:
                df.at[df.index[i], 'fvg_bullish'] = True
            # Bearish FVG: gap up then fill zone
            if c2['high'] < c1['low'] and c3['close'] < c2['high']:
                df.at[df.index[i], 'fvg_bearish'] = True
        return df
        
    def run(self) -> pd.DataFrame:
        """
        Orchestrates data fetching and all SMC computations.
        Returns enriched DataFrame.
        """
        df = self.fetch_data()
        df = self.find_swings(df)
        df = self.identify_structure_breaks(df)
        df = self.detect_order_blocks(df)
        df = self.detect_fvg(df)
        return df


if __name__ == "__main__":
    smc = SMCStrategy(symbol="SOLUSDT", interval="15m", limit=500)
    result = smc.run()
    print(result.tail(20))
