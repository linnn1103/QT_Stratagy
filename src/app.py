import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.fetch_data import DataFetcher
from strategy.structure import SMCStrategy
from utils.indicator import Indicator

@st.cache_data(show_spinner=False)
def get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul, slope_lookback) -> pd.DataFrame:
    fetcher = DataFetcher(symbol=symbol, interval=interval, limit=limit)
    df = fetcher.fetch_klines()
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


def main():
    st.title("ㄈㄈㄈㄈ")

    # Sidebar parameters
    st.sidebar.header("參數設定")
    symbol = st.sidebar.text_input("交易對", value="SOLUSDT")
    interval = st.sidebar.selectbox(
        "時間框架", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2
    )
    limit = st.sidebar.slider(
        "資料筆數", min_value=100, max_value=2000, value=600, step=100
    )
    atr_period = st.sidebar.number_input("ATR長度", min_value=1, max_value=100, value=14)
    swing_window = st.sidebar.number_input("擺動點長度", min_value=1, max_value=50, value=5)
    fvg_window = st.sidebar.number_input("FVG長度", min_value=2, max_value=50, value=3)
    vol_mul = st.sidebar.slider("交易量過濾倍數", 0.1, 5.0, value=1.3, step=0.1)
    slope_lookback = st.sidebar.number_input("ADX斜率長度", min_value=2, max_value=100, value=14)
    atr_filter = st.sidebar.number_input("ATR過濾數值", min_value=0.0, max_value=5.0, value=0.7, step=0.1)
    adx_slope_filter = st.sidebar.number_input("ADX斜率過濾", min_value=-1.0, max_value=0.0, value=-0.1, step=0.01)
    stop_loss_factor = st.sidebar.number_input("止損倍數", min_value=0.0, max_value=1.0, value=0.035, step=0.005)
    with st.spinner("Fetching data and computing strategy..."):
        df = get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul, slope_lookback)
    st.success("Computation completed!")

    # Prepare trade lists
    open_trades = []
    closed_trades = []

    # Iterate over each bar to handle entry and exit sequentially
    for idx, row in df.iterrows():
        # First: check exit signals for any open trade
        for trade in list(open_trades):
            # exit if reverse CHOCH + ATR > threshold
            if row['timestamp'] > trade['entry_time'] and row['CHOCH'] and row['ATR'] > atr_filter:
                # confirm reverse direction
                if (trade['direction'] == 'Long' and row['close'] < row['open']) or \
                   (trade['direction'] == 'Short' and row['close'] > row['open']):
                    trade['exit_time'] = row['timestamp']
                    trade['exit_price'] = row['close']
                    trade['is_closed'] = True
                    closed_trades.append(trade)
                    open_trades.remove(trade)
        # Then: check entry conditions
        if not row['CHOCH']:
            continue
        ADX_slope = row.get('ADX_slope', np.nan)
        if np.isnan(ADX_slope) or ADX_slope >= adx_slope_filter:
            continue
        direction = 'Long' if row['close'] > row['open'] else 'Short'
        # allow entry only if no open or same direction
        if open_trades and open_trades[0]['direction'] != direction:
            continue
        entry_time = row['timestamp']
        entry_price = row['close']
        if direction == 'Long':
            prev = df[df['structure_break'] == 'break_low'].loc[:idx-1]
            if not prev.empty:
                stop_loss = prev.iloc[-1]['low']
            else:
                stop_loss = entry_price * (1 - stop_loss_factor)  
        else:
            prev = df[df['structure_break'] == 'break_high'].loc[:idx-1]
            if not prev.empty:
                stop_loss = prev.iloc[-1]['high']
            else:
                stop_loss = entry_price * (1 + stop_loss_factor)
        open_trades.append({
            'entry_time': entry_time,
            'entry_price': entry_price,
            'direction': direction,
            'stop_loss': stop_loss,
            'is_closed': False
        })

    # Display ongoing and closed trades
    if open_trades:
        st.subheader("Open Trades")
        st.dataframe(pd.DataFrame(open_trades))
    if closed_trades:
        closed_df = pd.DataFrame(closed_trades)
        closed_df['pnl'] = (closed_df['entry_price'] - closed_df['exit_price']) * closed_df['direction'].apply(lambda x: 1 if x == 'Long' else -1)
        st.subheader("Closed Trades")
        st.dataframe(closed_df)

    # Build dynamic zones
    zones = {'ob': [], 'fvg': [], 'structure': []}
    last_time = df['timestamp'].iat[-1]

    # Order Block, FVG, Structure zones (logic unchanged)
    for i, row in df.iterrows():
        if row['bullish_ob']:
            y0, y1 = row['low'], row['high']
            broken = any(df['close'].iloc[j] <= y0 for j in range(i+1, len(df)))
            if not broken:
                zones['ob'].append({'type': 'bullish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
        if row['bearish_ob']:
            y0, y1 = row['low'], row['high']
            broken = any(df['close'].iloc[j] >= y1 for j in range(i+1, len(df)))
            if not broken:
                zones['ob'].append({'type': 'bearish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
    for i in range(fvg_window, len(df)):
        if df['fvg_bullish'].iat[i]:
            c1, c2 = df.iloc[i-fvg_window], df.iloc[i-fvg_window+1]
            y0, y1 = c1['high'], c2['low']
            broken = any(df['close'].iloc[j] <= y1 for j in range(i+1, len(df)))
            if not broken:
                zones['fvg'].append({'type': 'bullish', 'start': c2['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
        if df['fvg_bearish'].iat[i]:
            c1, c2 = df.iloc[i-fvg_window], df.iloc[i-fvg_window+1]
            y0, y1 = c2['high'], c1['low']
            broken = any(df['close'].iloc[j] >= y0 for j in range(i+1, len(df)))
            if not broken:
                zones['fvg'].append({'type': 'bearish', 'start': c2['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
    for i, row in df.iterrows():
        if row['structure_break'] == 'break_high':
            y0, y1 = row['low'], row['high']
            broken = any(df['close'].iloc[j] <= y0 for j in range(i+1, len(df)))
            if not broken:
                zones['structure'].append({'type': 'bullish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
        if row['structure_break'] == 'break_low':
            y0, y1 = row['low'], row['high']
            broken = any(df['close'].iloc[j] >= y1 for j in range(i+1, len(df)))
            if not broken:
                zones['structure'].append({'type': 'bearish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC', customdata=df['ADX'],hoverinfo='text', text=df['ADX'].apply(lambda x: f"ADX: {x:.2f}"), showlegend=False
    ))
    # Swing markers
    fig.add_trace(go.Scatter(
        x=df[df['swing_high']]['timestamp'], y=df[df['swing_high']]['high'], mode='markers', name='Swing High', marker_symbol='triangle-up', marker_size=10
    ))
    fig.add_trace(go.Scatter(
        x=df[df['swing_low']]['timestamp'], y=df[df['swing_low']]['low'], mode='markers', name='Swing Low', marker_symbol='triangle-down', marker_size=10
    ))
    # BOS & CHOCH markers
    fig.add_trace(go.Scatter(
        x=df[df['BOS']]['timestamp'], y=df[df['BOS']]['close'], mode='markers', name='BOS', marker_symbol='x', marker_size=12
    ))
    fig.add_trace(go.Scatter(
        x=df[df['CHOCH']]['timestamp'], y=df[df['CHOCH']]['close'], mode='markers', name='CHOCH', marker_symbol='diamond', marker_size=12
    ))
    # Zones shapes
    for ob in zones['ob']:
        color = 'rgba(0,255,0,0.2)' if ob['type']=='bullish' else 'rgba(255,0,0,0.2)'
        fig.add_shape(type='rect', x0=ob['start'], x1=ob['end'], y0=ob['y0'], y1=ob['y1'], fillcolor=color, line_width=0, layer='below')
    for fvg in zones['fvg']:
        color = 'rgba(0,200,0,0.15)' if fvg['type']=='bullish' else 'rgba(200,0,0,0.15)'
        fig.add_shape(type='rect', x0=fvg['start'], x1=fvg['end'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=color, line_width=0, layer='below')
    for stz in zones['structure']:
        color = 'rgba(0,0,255,0.2)' if stz['type']=='bullish' else 'rgba(255,165,0,0.2)'
        fig.add_shape(type='rect', x0=stz['start'], x1=stz['end'], y0=stz['y0'], y1=stz['y1'], fillcolor=color, line_width=0, layer='below')

    fig.update_layout(
        title=f"{symbol} {interval} Reactive Chart with SMC Zones & Accurate Breaks", xaxis_title='Time', yaxis_title='Price', hovermode='x unified', dragmode='zoom', yaxis=dict(fixedrange=False)
    )
    st.plotly_chart(
        fig, use_container_width=True, config={'scrollZoom':True,'displayModeBar':True,'editable':False}
    )

    # Signals table
    st.subheader("Detected Signals & Zones")
    display_cols = ['timestamp','structure_break','BOS','CHOCH','bullish_ob','bearish_ob','fvg_bullish','fvg_bearish']
    st.dataframe(df[display_cols].tail(50))

if __name__ == '__main__':
    main()