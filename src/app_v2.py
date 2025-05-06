import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from strategy.structure import SMCStrategy
from data.websocket_fetch import WebSocketDataFetcher
from streamlit import st_autorefresh

# Initialize session state for live data
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=[
        'timestamp','open','high','low','close','volume','close_time','quote_volume'
    ])
if 'ws' not in st.session_state:
    # callback to append new candles
    def on_new_candle(candle: dict):
        df = st.session_state.df
        st.session_state.df = df.append(candle, ignore_index=True)

# Start WebSocket on first run
if 'ws_started' not in st.session_state:
    fetcher = WebSocketDataFetcher(
        symbol=st.sidebar.text_input("Symbol", "SOLUSDT"),
        interval=st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","4h"], index=2),
        on_candle=on_new_candle
    )
    fetcher.start()
    st.session_state.ws = fetcher
    st.session_state.ws_started = True

# Auto-refresh every second
st_autorefresh(interval=1000, key="live_update")

# Main parameters (cached)
symbol = st.sidebar.text_input("Symbol", value="SOLUSDT")
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","30m","1h","4h"], index=2)
limit = st.sidebar.slider("Data Points (limit)", 100, 2000, 500, 100)
atr_period = st.sidebar.number_input("ATR Period", 1, 100, 14)
swing_window = st.sidebar.number_input("Swing Window", 1, 50, 14)
fvg_window = st.sidebar.number_input("FVG Window", 2, 50, 3)
vol_mul = st.sidebar.slider("Volatility Multiplier", 0.1, 5.0, 1.5, 0.1)

# Prepare live DataFrame
df_live = st.session_state.df.copy()
# Trim to latest 'limit' entries
if len(df_live) > limit:
    df_live = df_live.iloc[-limit:].reset_index(drop=True)
# Convert time
df_live['timestamp'] = (
    pd.to_datetime(df_live['timestamp'])
    .dt.tz_localize('UTC').dt.tz_convert('Asia/Taipei')
    .dt.tz_localize(None)
)

# Run SMCStrategy on live data
if len(df_live) >= swing_window*2:
    smc = SMCStrategy(
        symbol=symbol, interval=interval, limit=len(df_live),
        atr_period=atr_period, swing_window=swing_window,
        fvg_window=fvg_window, volatility_multiplier=vol_mul
    )
    enriched = smc.run_from_df(df_live)  # assumes you add run_from_df
else:
    enriched = df_live

# Plot
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=enriched['timestamp'], open=enriched['open'], high=enriched['high'],
    low=enriched['low'], close=enriched['close'], name='OHLC'
))
# add markers and zones as in previous app...

st.plotly_chart(fig, use_container_width=True, config={'scrollZoom':True,'displayModeBar':True})

# Signals table
display_cols = ['timestamp','structure_break','BOS','CHOCH','bullish_ob','bearish_ob','fvg_bullish','fvg_bearish']
st.dataframe(enriched[display_cols].tail(50))
