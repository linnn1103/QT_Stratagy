import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from strategy.structure import SMCStrategy

# -- Streamlit Visualization for SMC Strategy with Reactive Zones & Accurate Break Logic --

@st.cache_data(show_spinner=False)
def get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul):
    smc = SMCStrategy(
        symbol=symbol,
        interval=interval,
        limit=limit,
        atr_period=atr_period,
        swing_window=swing_window,
        fvg_window=fvg_window,
        volatility_multiplier=vol_mul
    )
    return smc.run()


def main():
    st.title("SMC Strategy Dashboard (Reactive)")

    # Sidebar parameters
    st.sidebar.header("Parameters")
    symbol = st.sidebar.text_input("Symbol", value="SOLUSDT")
    interval = st.sidebar.selectbox(
        "Interval", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=2
    )
    limit = st.sidebar.slider(
        "Data Points (limit)", min_value=100, max_value=2000, value=500, step=100
    )
    atr_period = st.sidebar.number_input(
        "ATR Period", min_value=1, max_value=100, value=14
    )
    swing_window = st.sidebar.number_input(
        "Swing Window", min_value=1, max_value=50, value=30
    )
    fvg_window = st.sidebar.number_input(
        "FVG Window", min_value=2, max_value=50, value=21
    )
    vol_mul = st.sidebar.slider(
        "Volatility Multiplier", 0.1, 5.0, value=1.5, step=0.1
    )

    # Reactive compute on change
    with st.spinner("Fetching data and computing strategy..."):
        df = get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul)
    st.success("Computation completed!")

    # Identify dynamic OB, FVG, Structure zones
    zones = {'ob': [], 'fvg': [], 'structure': []}
    last_time = df['timestamp'].iat[-1]

    # Order Block zones with accurate break detection (close-based)
    for i, row in df.iterrows():
        if row['bullish_ob']:
            y0, y1 = row['low'], row['high']
            # breakout when a future close <= bottom of the zone
            broken = any(df['close'].iloc[j] <= y0 for j in range(i+1, len(df)))
            if not broken:
                zones['ob'].append({'type': 'bullish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
        if row['bearish_ob']:
            y0, y1 = row['low'], row['high']
            # breakout when a future close >= top of the zone
            broken = any(df['close'].iloc[j] >= y1 for j in range(i+1, len(df)))
            if not broken:
                zones['ob'].append({'type': 'bearish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})

    # FVG zones with close-based break detection
    n = fvg_window
    for i in range(n, len(df)):
        if df['fvg_bullish'].iat[i]:
            c1, c2 = df.iloc[i-n], df.iloc[i-n+1]
            y0, y1 = c1['high'], c2['low']
            broken = any(df['close'].iloc[j] <= y1 for j in range(i+1, len(df)))
            if not broken:
                zones['fvg'].append({'type': 'bullish', 'start': c2['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
        if df['fvg_bearish'].iat[i]:
            c1, c2 = df.iloc[i-n], df.iloc[i-n+1]
            y0, y1 = c2['high'], c1['low']
            broken = any(df['close'].iloc[j] >= y0 for j in range(i+1, len(df)))
            if not broken:
                zones['fvg'].append({'type': 'bearish', 'start': c2['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})

    # Structure break zones: draw original structure ranges until a close confirms reset
    for i, row in df.iterrows():
        if row['structure_break'] in ['break_high', 'break_low']:
            y0, y1 = row['low'], row['high']
            if row['structure_break'] == 'break_high':
                # bullish structure: breakout reset when close <= low
                broken = any(df['close'].iloc[j] <= y0 for j in range(i+1, len(df)))
                if not broken:
                    zones['structure'].append({'type': 'bullish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})
            else:
                # bearish structure: breakout reset when close >= high
                broken = any(df['close'].iloc[j] >= y1 for j in range(i+1, len(df)))
                if not broken:
                    zones['structure'].append({'type': 'bearish', 'start': row['timestamp'], 'end': last_time, 'y0': y0, 'y1': y1})

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC'
    ))
    # Swings
    fig.add_trace(go.Scatter(
        x=df[df['swing_high']]['timestamp'], y=df[df['swing_high']]['high'], mode='markers',
        name='Swing High', marker_symbol='triangle-up', marker_size=10
    ))
    fig.add_trace(go.Scatter(
        x=df[df['swing_low']]['timestamp'], y=df[df['swing_low']]['low'], mode='markers',
        name='Swing Low', marker_symbol='triangle-down', marker_size=10
    ))
    # Draw zones
    for ob in zones['ob']:
        color = 'rgba(0,255,0,0.2)' if ob['type'] == 'bullish' else 'rgba(255,0,0,0.2)'
        fig.add_shape(type='rect', x0=ob['start'], x1=ob['end'], y0=ob['y0'], y1=ob['y1'], fillcolor=color, line_width=0, layer='below')
    for fvg in zones['fvg']:
        color = 'rgba(0,200,0,0.15)' if fvg['type'] == 'bullish' else 'rgba(200,0,0,0.15)'
        fig.add_shape(type='rect', x0=fvg['start'], x1=fvg['end'], y0=fvg['y0'], y1=fvg['y1'], fillcolor=color, line_width=0, layer='below')
    for stz in zones['structure']:
        color = 'rgba(0,0,255,0.2)' if stz['type'] == 'bullish' else 'rgba(255,165,0,0.2)'
        fig.add_shape(type='rect', x0=stz['start'], x1=stz['end'], y0=stz['y0'], y1=stz['y1'], fillcolor=color, line_width=0, layer='below')

    fig.update_layout(
        title=f"{symbol} {interval} Reactive Chart with SMC Zones & Accurate Breaks",
        xaxis_title='Time', yaxis_title='Price', hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)

    # Signals table
    st.subheader("Detected Signals & Zones")
    display_cols = ['timestamp', 'structure_break', 'bullish_ob', 'bearish_ob', 'fvg_bullish', 'fvg_bearish']
    st.dataframe(df[display_cols].tail(50))

if __name__ == '__main__':
    main()
