import talib
import pandas as pd
class Indicator:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_EMA(self, period=30) -> pd.Series:
        """
        Calculate the Exponential Moving Average (EMA) for the given period.
        :param period: The period for the EMA calculation.
        """
        close_prices = self.data['close'].values.astype(float)
        return talib.EMA(close_prices, timeperiod=period)

    def get_ADX(self, period=14) -> pd.Series:
        """
        Calculate the Average Directional Index (ADX) for the given period.
        :param period: The period for the ADX calculation.
        """
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        return talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)

    def get_DI(self, period=14) -> tuple:
        """
        Calculate the Directional Indicators (DI) for the given period.
        :param period: The period for the DI calculation.
        """
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        plus_di = talib.PLUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        minus_di = talib.MINUS_DI(high_prices, low_prices, close_prices, timeperiod=period)
        return plus_di, minus_di

    def get_ATR(self, period=14) -> pd.Series:
        """
        Calculate the Average True Range (ATR) for the given period.
        :param period: The period for the ATR calculation.
        """
        high_prices = self.data['high'].values.astype(float)
        low_prices = self.data['low'].values.astype(float)
        close_prices = self.data['close'].values.astype(float)
        return talib.ATR(high_prices, low_prices, close_prices,timeperiod=period)