import data.fetch_data as fetch_data
import utils.indicator as indicator
import numpy as np
INTERVAL = ['1m', '15m', '1h', '4h']
INDICATORS = ['EMA', 'ADX', '+DI', '-DI']

def get_data():
    klines = {}
    calculatedIndicators = {}
    for interval in INTERVAL:
        data_fetcher = fetch_data.DataFetcher(interval=interval)
        fetched_klines = data_fetcher.fetch_klines()  
        if fetched_klines:  
            klines[interval] = fetched_klines
        else:
            klines[interval] = [] 
    for interval in INTERVAL:
        if klines[interval]:
            calculatedIndicators[interval] = {}
            indicator_calculator = indicator.Indicator(klines[interval])
            for ind in INDICATORS:
                if ind == 'EMA':
                    calculatedIndicators[interval]['EMA'] = indicator_calculator.get_EMA(9)
                elif ind == 'ADX':
                    calculatedIndicators[interval]['ADX'] = indicator_calculator.get_ADX(14)
                elif ind == '+DI':
                    calculatedIndicators[interval]['+DI'], _ = indicator_calculator.get_DI(14)
                elif ind == '-DI':
                    _, calculatedIndicators[interval]['-DI'] = indicator_calculator.get_DI(14)

    return klines, calculatedIndicators

def analyze_trend(indicator, period=4):
    """
    計算 ADX 的回歸曲線斜率
    period = 4
    """
    adx_values = indicator['ADX'][-period:]

    x = np.arange(len(adx_values))

    slope, _ = np.polyfit(x, adx_values, 1)  
    print(f"-DI: {indicator['-DI'][-1]}")
    print(f"+DI: {indicator['+DI'][-1]}")
    print(f"ADX: {adx_values[-1]}")
    print(f"Slope: {slope}")
    if abs(slope) >= 0.2 and adx_values[-1] > 20:
        trend = "Uptrend" if indicator['+DI'][-1] > indicator['-DI'][-1] else "Downtrend"

    else:
        trend = "No Trend"
    return trend

if __name__ == "__main__":
    klines, calculatedIndicators = get_data()
    s = analyze_trend(calculatedIndicators['15m'])
    print(s)