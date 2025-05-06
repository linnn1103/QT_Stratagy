from utils.indicator import Indicator
from data.fetch_data import DataFetcher

class Filter:
    def __init__(self,interval):
        self.df = DataFetcher(interval=interval).fetch_klines()
        print(f"Data fetched for interval: {interval}")
        
    def ADX_fillter(self,idx:int) -> bool:
        result = True if Indicator(self.df).get_ADX()[idx] < 18 else False
        return result   