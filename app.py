import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Liquidity Sniper Pro", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGGING ENGINE ---
LOG_FILE = "sniper_logs.csv"
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Time,Symbol,Price,OFI,Signal,Risk_Price\n")

def log_trade(symbol, price, ofi, signal, risk_price):
    init_log()
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{ofi:.3f},{signal},{risk_price}\n")

init_log()

# --- 3. DATA ENGINE ---
def get_orderbook(sym):
    try:
        url = f"https://api.exchange.coinbase.com/products/{sym}/book?level=2"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        return requests.get(url, headers=headers, timeout=5).json()
    except: return None

def get_candles(sym, granularity=300):
    try:
        url = f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity={granularity}"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        df = pd.DataFrame(res, columns=["time", "low", "high", "open", "close", "vol"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except: return pd.DataFrame()

# --- 4. MATH ENGINE ---
def calculate_indicators(df):
    if df.empty: return 50, 0, pd.Series()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    vwap_series = df['pv'].cumsum() / df['vol'].cumsum()
    
    return rsi.iloc[-1], vwap_series.iloc[-1], vwap_series

def get_real_volatility(symbol):
    if "vol_data" not in st.session_state or time.time() - st.session_state.vol_time > 60:
        df = get_candles(symbol, granularity=86400)
        if df.empty: return 0.05
        returns = df['close'].pct_change().dropna()
        st.session_state.vol_data = returns.std()
        st.session_state.vol_time = time.time()
    return st.session_state.vol_data

def run_monte_carlo(current_price, volatility, simulations=1000):
    daily_return = np.random.normal(0, volatility, simulations)
    future_prices = current_price * (1 + daily_return)
    future_prices = np.sort(future_prices)
    var_price = np.percentile(future_prices, 5) 
    return var_price

def calculate_ofi(bids, asks):
    if not bids: return 0
    b_vol = sum([float(x[1]) for x in bids])
    a_vol = sum([float(x[1]) for x in asks])
    return (b_vol - a_vol) / (b_vol + a_vol)

def get_walls(orders, price):
    walls = []
    threshold = 50000 if price > 1000 else 10000 
    for order in orders:
        p = float(order[0])
        s = float(order[1])
        val = p * s
        if val > threshold:
            walls.append((p, val)) 
    return walls[:3]

# --- 5. VISUAL CHARTING ENGINE (UPDATED) ---
def plot_chart(df, vwap_series, symbol, stop_loss, buy_walls, sell_walls):
    fig = go.Figure()

    # 1. Candlesticks
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name="Price"
    ))

    # 2. VWAP
    fig.add_trace(go.Scatter(
        x=df['time'], y=vwap_series,
        mode='lines', name='VWAP',
        line=dict(color='orange', width=2)
    ))
    
    # 3. Stop Loss (Monte Carlo)
    fig.add_hline(y=stop_loss, line_dash="dot", 
                  annotation_text="🛑 VaR Risk Line", 
                  annotation_position="bottom right",
                  line_color="#FF00FF", line_width=2)

    # 4. Buy Walls (Green Lines)
    for p, v in buy_walls:
        fig.add_hline(y=p, line_color="#00FF00", opacity=0.4, 
                      annotation_text=f"🐳 Buy ${v/1000:.0f}k",
                      annotation_position="bottom right")

    # 5. Sell Walls (Red Lines)
    for p, v in sell_walls:
        fig.add_hline(y=p, line_color="#FF0000", opacity=0.4,
                      annotation_text=f"🐳 Sell ${v/1000:.0f}k",
                      annotation_position="top right")

    # Layout
    fig.update_layout(
        title=f"{symbol} Visual Command Center",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

# --- 6. APP LOGIC ---
st.sidebar.header("⚙️ Sniper Scope")
sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()
MAP = {"GOLD": "PAXG-USD", "XAUUSD": "PAXG-USD", "BITCOIN": "BTC-USD"}
symbol = MAP.get(sym_input, sym_input)
if "-" not in symbol and len(symbol)>3: symbol = f"{symbol[:-3]}-{symbol[-3:]}"
st.sidebar.markdown("---")
trade_size = st.sidebar.number_input("My Trade Size ($)", value=90.0, step=10.0)

st.title(f"🦅 Liquidity Sniper: {symbol}")

# GET DATA
book_data = get_orderbook(symbol)
candle_data = get_candles(symbol, 300)

if not book_data or candle_data.empty:
    st.error("Waiting for data...")
    time.sleep(1)
    st.rerun()

bids = book_data['bids']
asks = book_data['asks']
price = float(bids[0][0])
ofi = calculate_ofi(bids, asks)
rsi, vwap_val, vwap_series = calculate_indicators(candle_data)

# WALLS
buy_walls = get_walls(bids, price)
sell_walls = get_walls(asks, price)

# RISK CALCULATION
vol = get_real_volatility(symbol)
stop_loss_price = run_monte_carlo(price, vol)
percent_drop = (price - stop_loss_price) / price
dollar_risk = trade_size * percent_drop

# SIGNALS
signal = "WAIT"
score = 0
reasons = []

if ofi > 0.15: score += 1; reasons.append("Aggressive Buying")
elif ofi < -0.15: score -= 1; reasons.append("Aggressive Selling")

if 40 < rsi < 70: 
    if score > 0: score += 1
    elif score < 0: score -= 1
else: reasons.append(f"RSI Risky ({rsi:.0f})")

if price > vwap_val: 
    if score > 0: score += 1
    reasons.append("Uptrend")
else: 
    if score < 0: score -= 1
    reasons.append("Downtrend")

if score >= 3: signal = "PERFECT BUY 🟢"
elif score <= -3: signal = "PERFECT SELL 🔴"

# AUTO LOG
if "PERFECT" in signal and signal != st.session_state.get('last_sig', ''):
    log_trade(symbol, price, ofi, signal, f"${stop_loss_price:.2f}")
    st.session_state.last_sig = signal
    st.toast("Trade Logged!")

# --- DISPLAY METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${price:,.2f}")
col2.metric("OFI Pressure", f"{ofi:.3f}")
col3.metric("RSI (14)", f"{rsi:.1f}")

# --- THE SUPER CHART ---
# Passing all the data into the visualizer
chart = plot_chart(candle_data, vwap_series, symbol, stop_loss_price, buy_walls, sell_walls)
st.plotly_chart(chart, use_container_width=True)

st.divider()

sc1, sc2 = st.columns([1.5, 1])
with sc1:
    st.subheader(f"🎯 Signal: {signal}")
    st.write(f"**Confluence Score:** {abs(score)}/3")
    for r in reasons: st.caption(f"• {r}")
with sc2:
    st.subheader("🎲 Risk Monitor")
    st.write(f"**Stop Loss:** ${stop_loss_price:,.2f}")
    risk_color = "green" if dollar_risk < (trade_size * 0.02) else "red"
    st.markdown(f"**Risk:** <span style='color:{risk_color}; font-size:24px'>-${dollar_risk:.2f}</span>", unsafe_allow_html=True)

time.sleep(1)
st.rerun()
