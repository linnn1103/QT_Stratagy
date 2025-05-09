import talib
import pandas as pd
class Indicator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_EMA(self, period=30) -> pd.Series:
        close_prices = self.data['close'].values.astype(float)
        return talib.EMA(close_prices, timeperiod=period)

    def get_ADX(self, period=14) -> pd.Series:
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        return talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)

    def get_DI(self, period=14) -> tuple:
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        return plus_di, minus_di

    def get_ATR(self, period=14) -> pd.Series:
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        return talib.ATR(high_prices, low_prices, close_prices,timeperiod=period)