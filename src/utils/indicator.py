import talib
import numpy as np
import pandas as pd
class Indicator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def get_EMA(self, period: int) -> pd.Series:
        close_prices = np.array([d['close'] for d in self.data], dtype=float)
        return talib.EMA(close_prices, timeperiod=period)
    
    def get_ADX(self, period: int) -> pd.Series:
        high_prices = np.array([d['high'] for d in self.data], dtype=float)
        low_prices = np.array([d['low'] for d in self.data], dtype=float)
        close_prices = np.array([d['close'] for d in self.data], dtype=float)
        return talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
    
    def get_DI(self, period: int) -> tuple:
        high_prices = np.array([d['high'] for d in self.data], dtype=float)
        low_prices = np.array([d['low'] for d in self.data], dtype=float)
        close_prices = np.array([d['close'] for d in self.data], dtype=float)
        plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        return plus_di, minus_di
    