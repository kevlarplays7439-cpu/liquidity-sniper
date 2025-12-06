import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import argrelextrema
import os

# --- 1. CONFIG & CSS (THE BEAUTY TREATMENT) ---
st.set_page_config(page_title="Liquidity Sniper Pro", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    /* Global Clean Look */
    .block-container { padding-top: 1rem; max-width: 95%; }
    
    /* Card Styling */
    .stContainer {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Green/Red Tags */
    .signal-box {
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGGING ---
LOG_FILE = "sniper_logs.csv"
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Time,Symbol,Price,Signal,Risk_SL\n")

def log_trade(symbol, price, signal, risk_sl):
    init_log()
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{signal},{risk_sl}\n")

init_log()

# --- 3. DATA ENGINE ---
def fetch_data(sym):
    try:
        headers = {"User-Agent": "LiquidityLens/1.0"}
        candles = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity=60", headers=headers, timeout=2).json()
        book = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/book?level=2", headers=headers, timeout=2).json()
        return book, candles
    except: return None, None

# --- 4. MATH ENGINE ---
def analyze_market(book_res, candle_res):
    if not book_res or not candle_res: return None
    
    # Data Prep
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values("time").reset_index(drop=True)
    
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    last_idx = df.index[-1]
    df.at[last_idx, 'close'] = price
    
    # Indicators
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    df['vwap'] = df['pv'].cumsum() / df['vol'].cumsum()

    # OFI
    b_vol = sum([float(x[1]) for x in bids])
    a_vol = sum([float(x[1]) for x in asks])
    ofi = (b_vol - a_vol) / (b_vol + a_vol) if (b_vol+a_vol) > 0 else 0

    # Volatility & Risk
    returns = df['close'].pct_change().dropna()
    vol_1min = returns.std()
    vol_hour = vol_1min * np.sqrt(60)
    daily_return = np.random.normal(0, vol_hour, 2000)
    future_prices = price * (1 + daily_return)
    stop_loss = np.percentile(np.sort(future_prices), 5)

    return {
        "price": price, "df": df, "ofi": ofi, "stop_loss": stop_loss,
        "rsi": df['rsi'].iloc[-1], "vwap": df['vwap'].iloc[-1], "ema": df['ema_50'].iloc[-1],
        "bids": bids, "asks": asks
    }

# --- 5. CHART ENGINE ---
def plot_chart(data):
    df = data['df']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], line=dict(color='#ff9800', width=2), name="VWAP"), row=1, col=1)
    fig.add_hline(y=data['stop_loss'], line_dash="dot", line_color="#d500f9", annotation_text="VaR Risk", row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], line=dict(color='#7e57c2', width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="gray", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="gray", dash="dot"), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, uirevision='TheTruth')
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("⚙️ Sniper Scope")
sym = st.sidebar.text_input("Symbol", "BTC-USD").upper()
if "-" not in sym and len(sym)>3: sym = f"{sym[:-3]}-{sym[-3:]}"
trade_size = st.sidebar.number_input("Trade Size ($)", value=90.0, step=10.0)

if st.sidebar.button("🛠️ Test Log"):
    log_trade(sym, 0, "TEST", "$0.00")
    st.toast("Row Added")

st.title(f"🦅 {sym} Terminal")

@st.fragment(run_every=1)
def main_loop():
    book, candle = fetch_data(sym)
    data = analyze_market(book, candle)
    if not data:
        st.caption("📡 Connecting...")
        return

    # Signal Logic
    score = 0
    if data['price'] > data['ema']: score += 1
    else: score -= 1
    if data['ofi'] > 0.15: score += 1
    elif data['ofi'] < -0.15: score -= 1
    if 40 < data['rsi'] < 60: pass # Neutral
    elif data['rsi'] >= 60: score += 1
    elif data['rsi'] <= 40: score -= 1

    signal = "WAIT"
    sig_color = "#555" # Grey
    if score >= 3: 
        signal = "STRONG BUY"
        sig_color = "#00c853" # Green
    elif score <= -3: 
        signal = "STRONG SELL"
        sig_color = "#d50000" # Red

    # Risk Calc
    dollar_risk = trade_size * ((data['price'] - data['stop_loss']) / data['price'])

    # Auto Log
    if "STRONG" in signal and signal != st.session_state.get('last_sig', ''):
        log_trade(sym, data['price'], signal, f"${data['stop_loss']:.2f}")
        st.session_state.last_sig = signal
        st.toast(f"Signal: {signal}")

    # --- NEW CLEAN LAYOUT ---
    
    # 1. Top Metrics (Minimal)
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${data['price']:,.2f}")
    c2.metric("OFI Flow", f"{data['ofi']:.3f}")
    c3.metric("RSI", f"{data['rsi']:.1f}")

    # 2. Decision Engine (The Cards)
    c_sig, c_risk = st.columns([2, 1])
    
    with c_sig:
        st.markdown(f"""
        <div class="signal-box" style="background-color: {sig_color}; color: white;">
            {signal}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Confluence Score: {score}/3")

    with c_risk:
        with st.container(border=True):
            st.markdown("**Risk Monitor**")
            st.write(f"Stop Loss: **${data['stop_loss']:,.2f}**")
            risk_color = "red" if dollar_risk > (trade_size * 0.02) else "green"
            st.markdown(f"Risk: <span style='color:{risk_color}'>-${dollar_risk:.2f}</span>", unsafe_allow_html=True)

    # 3. The Chart (Clean)
    st.plotly_chart(plot_chart(data), use_container_width=True)

    # 4. Hidden Log (The Drawer)
    with st.expander("📝 Trade History"):
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE).tail(5), use_container_width=True, hide_index=True)

main_loop()
