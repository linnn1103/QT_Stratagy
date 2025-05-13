import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data.fetch_data import DataFetcher
from strategy.structure import SMCStrategy
from utils.indicator import Indicator
from streamlit_autorefresh import st_autorefresh
from trade.order import OrderLogic
from backtest.backtest import BacktestDataSaver
# import time, datetime
import json, os
TRADES_FILE = "trades_record.json"

@st.cache_data(show_spinner=False)

def save_trades(open_trades, closed_trades):
    """
    Save open and closed trades to a JSON file.
    :param open_trades: list, list of open trades
    :param closed_trades: list, list of closed trades
    """
    data = {
        "open_trades": open_trades,
        "closed_trades": closed_trades
    }
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, default=str, indent=2)

def load_trades():
    """
    Load open and closed trades from a JSON file.
    """
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("open_trades", []), data.get("closed_trades", [])
    return [], []

def close_trade(trade, close_type, close_price, close_time) -> dict:
    """
    Closes a trade and logs the details.
    :param trade: dict, the trade to be closed
    :param close_type: str, the type of closing (e.g., '止損', '止盈')
    :param close_price: float, the price at which the trade is closed
    :param close_time: datetime, the time at which the trade is closed
    """
    print(f"已平倉: {trade['direction']} 價格: {close_price} 時間: {close_time} 平倉類型: {close_type}")
    closed_trade = trade.copy()
    closed_trade['exit_price'] = close_price
    closed_trade['exit_time'] = close_time
    closed_trade['close_type'] = close_type
    closed_trade['is_closed'] = True
    return closed_trade

def get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul, slope_lookback) -> pd.DataFrame:
    """
    Fetches data and computes the SMC strategy indicators.
    :param symbol: str, the trading pair symbol
    :param interval: str, the time interval for the data
    :param limit: int, the number of data points to fetch
    :param atr_period: int, the period for ATR calculation
    :param swing_window: int, the window for swing point detection
    :param fvg_window: int, the window for FVG detection
    :param vol_mul: float, the multiplier for volume filtering
    :param slope_lookback: int, the lookback period for ADX slope calculation
    """
    fetcher = DataFetcher(symbol=symbol, interval=interval, limit=limit)
    df = fetcher.fetch_klines()

    ##########################回測用
    # a=BacktestDataSaver()
    # df = a.rr()
    ##########################
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
    # 強制初始化為 list，避免被覆蓋成 dict
    if not isinstance(st.session_state.get('open_trades', []), list):
        st.session_state.open_trades = []
    if not isinstance(st.session_state.get('closed_trades', []), list):
        st.session_state.closed_trades = []

    if 'open_trades' not in st.session_state or 'closed_trades' not in st.session_state:
        open_trades, closed_trades = load_trades()
        st.session_state.open_trades = open_trades
        st.session_state.closed_trades = closed_trades
        
    st_autorefresh(interval=10000, limit=0, key="data_refresh")
    st.title("ㄈㄈㄈㄈ")

    # Sidebar parameters
    st.sidebar.header("參數設定")
    symbol = st.sidebar.selectbox(
        "交易對", ["BTCUSDT", "ETHUSDT", "SOLUSDT"], index=2
    )
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
    a = {"BTCUSDT": 210.0, "ETHUSDT": 6.5, "SOLUSDT": 0.7}
    atr_filter = st.sidebar.number_input("ATR過濾數值", min_value=0.0, value=a[symbol], step=0.1)
    adx_slope_filter = st.sidebar.number_input("ADX斜率過濾", min_value=-1.0, max_value=0.0, value=-0.1, step=0.01)
    stop_loss_factor = st.sidebar.number_input("止損倍數", min_value=0.0, max_value=1.0, value=0.035, step=0.005)
    leverage = st.sidebar.number_input("槓桿倍數", min_value=1, max_value=50, value=5, step=1)
    with st.spinner("Fetching data and computing strategy..."):
        df = get_smc_df(symbol, interval, limit, atr_period, swing_window, fvg_window, vol_mul, slope_lookback)
    st.info(f"資料刷新時間：{pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M:%S')}")
    
    last_row = df.iloc[-1]

    if 'open_trades' not in st.session_state:
        st.session_state.open_trades = []
    if 'closed_trades' not in st.session_state:
        st.session_state.closed_trades = []

    # Status tracking
    for trade in list(st.session_state.open_trades):
        # 止損
        if trade['direction'] == 'Long':
            if last_row['low'] <= trade['stop_loss']:
                closed = close_trade(trade, '止損', trade['stop_loss'], last_row['timestamp'])
                st.session_state.closed_trades.append(closed)
                st.session_state.open_trades.remove(trade)
                save_trades(st.session_state.open_trades, st.session_state.closed_trades)
            elif last_row['CHOCH'] and last_row['ATR'] > atr_filter and last_row['close'] < last_row['open']:
                closed = close_trade(trade, '止盈', last_row['close'], last_row['timestamp'])
                st.session_state.closed_trades.append(closed)
                st.session_state.open_trades.remove(trade)
                save_trades(st.session_state.open_trades, st.session_state.closed_trades)
                OrderLogic().close_order(
                    instId={'BTCUSDT': 'BTC-USDT-SWAP', 'ETHUSDT': 'ETH-USDT-SWAP', 'SOLUSDT': 'SOL-USDT-SWAP'}[symbol],
                    clOrdId=f"{symbol}",
                    direction='Long',
                )
        else:
            if last_row['high'] >= trade['stop_loss']:
                closed = close_trade(trade, '止損', trade['stop_loss'], last_row['timestamp'])
                st.session_state.closed_trades.append(closed)
                st.session_state.open_trades.remove(trade)
                save_trades(st.session_state.open_trades, st.session_state.closed_trades)
            elif last_row['CHOCH'] and last_row['ATR'] > atr_filter and last_row['close'] > last_row['open']:
                closed = close_trade(trade, '止盈', last_row['close'], last_row['timestamp'])
                st.session_state.closed_trades.append(closed)
                st.session_state.open_trades.remove(trade)
                save_trades(st.session_state.open_trades, st.session_state.closed_trades)
                OrderLogic().close_order(
                    instId={'BTCUSDT': 'BTC-USDT-SWAP', 'ETHUSDT': 'ETH-USDT-SWAP', 'SOLUSDT': 'SOL-USDT-SWAP'}[symbol],
                    clOrdId=f"{symbol}",
                    direction='Short',
                )

    # Trade logic
    can_open = False
    direction = None
    ADX_slope = last_row.get('ADX_slope', np.nan)
    if last_row['CHOCH'] and not np.isnan(ADX_slope) and ADX_slope < adx_slope_filter:
        direction = 'Long' if last_row['close'] > last_row['open'] else 'Short'
        # Same direction only
        if not st.session_state.open_trades or st.session_state.open_trades[0]['direction'] == direction:
            can_open = True

    if can_open:
        entry_price = last_row['close']
        entry_time = last_row['timestamp']
        clOrdId = f"{symbol}{direction.lower()}"
        # 檢查是否已存在同 entry_time 或 clOrdId 的訂單
        already_open = any(
            (str(trade.get('entry_time')) == str(entry_time)) or (trade.get('clOrdId') == clOrdId)
            for trade in st.session_state.open_trades
        )
        if not already_open:
            if direction == 'Long':
                prev = df[df['structure_break'] == 'break_low'].iloc[:-1]
                stop_loss = prev.iloc[-1]['low'] if not prev.empty else entry_price * (1 - stop_loss_factor)
            else:
                prev = df[df['structure_break'] == 'break_high'].iloc[:-1]
                stop_loss = prev.iloc[-1]['high'] if not prev.empty else entry_price * (1 + stop_loss_factor)
            OrderLogic().create_order(
                instId={'BTCUSDT': 'BTC-USDT-SWAP', 'ETHUSDT': 'ETH-USDT-SWAP', 'SOLUSDT': 'SOL-USDT-SWAP'}[symbol],
                leverage=leverage,
                clOrdId=clOrdId,
                direction=direction,
                entry_price=entry_price,
                stop_loss_price=stop_loss
            )
            st.session_state.open_trades.append({
                'clOrdId': clOrdId,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'direction': direction,
                'stop_loss': stop_loss,
                'is_closed': False
            })
            save_trades(st.session_state.open_trades, st.session_state.closed_trades)


    # Trade history 
    if st.session_state.open_trades:
        st.subheader("持倉中")
        open_df = pd.DataFrame(st.session_state.open_trades)
        if 'entry_time' in open_df.columns:
            open_df['entry_time'] = open_df['entry_time'].astype(str)
        st.dataframe(open_df)
    if st.session_state.closed_trades:
        # 過濾掉 None 或缺少必要欄位的紀錄
        valid_trades = [
            t for t in st.session_state.closed_trades
            if t and 'entry_price' in t and 'exit_price' in t and 'direction' in t
        ]
        closed_df = pd.DataFrame(valid_trades)
        if not closed_df.empty:
            if 'entry_time' in closed_df.columns:
                closed_df['entry_time'] = closed_df['entry_time'].astype(str)
            if 'exit_time' in closed_df.columns:
                closed_df['exit_time'] = closed_df['exit_time'].astype(str)
            closed_df['pnl'] = (closed_df['entry_price'] - closed_df['exit_price']) * closed_df['direction'].apply(lambda x: 1 if x == 'Long' else -1)
            st.subheader("以平倉交易")
            st.dataframe(closed_df)


    # Build dynamic zones
    zones = {'ob': [], 'fvg': [], 'structure': []}
    last_time = df['timestamp'].iat[-1]

    # Order Block, FVG, Structure zones
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
    # while True:
    #     now = datetime.datetime.now()
    #     if now.minute % 15 == 0:
    #         main()
    #         break
    #     else:
    #         time.sleep(5)
