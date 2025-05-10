import streamlit as st

st.set_page_config(page_title="倉位計算器", layout="centered")
st.title("📈倉位計算器")

st.markdown("""
<style>
@media (max-width: 600px) {
    .stNumberInput, .stSelectbox { 
        min-width: 100% !important;
    }
}

.stMetric {
    padding: 1rem;
    background: #f8f9fa !important;
    border-radius: 10px;
    border: 1px solid #e6e6e6 !important;
    color: #333333 !important;
}

.stMetric label {
    color: #666666 !important;
    font-size: 0.9rem !important;
}

.stMetric div {
    color: #222222 !important;
    font-size: 1.4rem !important;
    font-weight: bold;
}

.stTextInput input, .stNumberInput input, .stSelectbox select {
    color: #ffffff !important;
}

.stAlert {
    background: #f8d7da !important;
    color: #721c24 !important;
    border-color: #f5c6cb !important;
}
</style>
""", unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        principal = st.number_input("💰 本金 (USDT)", value=1000.0, min_value=0.0, step=100.0, format="%.2f")
    with col2:
        loss_pct = st.number_input("📉 最大虧損 (%)", value=2.0, min_value=0.0, step=0.5, format="%.2f")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        entry_price = st.number_input("入場價", value=100.0, min_value=0.0, step=1.0, format="%.2f")
    with col2:
        stop_loss_price = st.number_input("止損價", value=95.0, min_value=0.0, step=1.0, format="%.2f")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        leverage = st.number_input("⚖️ 槓桿倍數", value=5.0, min_value=1.0, step=1.0, format="%.2f")
    with col2:
        fee_rate = st.number_input("💸 手續費率", value=0.05, min_value=0.0, step=0.01, format="%.4f")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        target_price = st.number_input("🎯 目標價", value=0.0, min_value=0.0, format="%.2f",
                                       help="選填內容，當RR與目標價同時存在時，優先根據目標價計算")
    with col2:
        rr_input = st.number_input("📊 風險回報比", value=0.0, min_value=0.0, format="%.2f",
                                   help="選填內容，當RR與目標價同時存在時，優先根據目標價計算")

def auto_detect_direction(entry, stop_loss):
    if entry == stop_loss:
        return "invalid", 0.0
    direction = "Long" if stop_loss < entry else "Short"
    risk_pct = (entry - stop_loss)/entry * (1 if direction == "Long" else -1)
    return direction, risk_pct

max_loss_total = principal * (loss_pct / 100)
direction, risk_pct = auto_detect_direction(entry_price, stop_loss_price)

if direction == "invalid":
    st.error("⚠️ 止損價不能等於入場價！")
    st.stop()

if risk_pct <= 0:
    st.error(f"⚠️ 價格關係異常！系統判斷方向為 {direction} 但風險計算為負值")
    st.stop()

try:
    fee_contribution = 2 * (fee_rate / 100)
    denominator = leverage * (abs(risk_pct) + fee_contribution)
    margin = max_loss_total / denominator
    notional = margin * leverage
    total_fee = notional * 2 * (fee_rate / 100)
    
    st.markdown("---")
    st.subheader("📊 倉位計算")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("保證金要求", f"{margin:.2f} USD")
        st.metric("持倉總額", f"{notional:.2f} USD")
    with col2:
        st.metric("最大可承受虧損", f"{max_loss_total:.2f} USD")
        st.metric("總手續費", f"{total_fee:.2f} USD")

except ZeroDivisionError:
    st.error("計算錯誤：風險率不能為零")
    st.stop()

if target_price > 0 or rr_input > 0:
    st.markdown("---")
    st.subheader("📈 收益預測")
    
    try:
        if target_price > 0:
            if (direction == "Long" and target_price <= entry_price) or \
               (direction == "Short" and target_price >= entry_price):
                st.warning("⚠️ 目標價設置與交易方向矛盾")
                st.stop()
            
            reward_pct = abs(target_price - entry_price)/entry_price
            rr = reward_pct / abs(risk_pct)
        else:
            rr = rr_input
            reward_pct = abs(risk_pct) * rr
            target_price = entry_price * (1 + reward_pct) if direction == "Long" else entry_price * (1 - reward_pct)

        gross_profit = notional * reward_pct
        net_profit = gross_profit - total_fee

        col1, col2 = st.columns(2)
        with col1:
            st.metric("預測目標價", f"{target_price:.2f} USD")
            st.metric("預期回報率", f"{reward_pct*100:.2f}%")
        with col2:
            st.metric("風險回報比", f"{rr:.2f}:1")
            st.metric("淨收益", f"{net_profit:.2f} USD")

        if rr < 1:
            st.warning("⚠️ 風險回報比低於1:1，建議重新評估交易策略")
        elif net_profit < 0:
            st.error("❌ 淨收益為負值！手續費超過預期收益")

    except Exception as e:
        st.error(f"收益計算錯誤：{str(e)}")

st.markdown("---")