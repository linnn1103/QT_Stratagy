from utils.indicator import Indicator
class Filter:
    def __init__(self,df):
       self.df = df
       
    def ADX_fillter(self,idx:int) -> bool:
        result = True if Indicator(self.df).get_ADX()[idx] < 18 else False
        return result   