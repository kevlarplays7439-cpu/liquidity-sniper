import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Liquidity Sniper Pro", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; }
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGGING ---
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
@st.cache_data(ttl=1)
def fetch_data(sym):
    try:
        headers = {"User-Agent": "LiquidityLens/1.0"}
        book_res = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/book?level=2", headers=headers, timeout=5).json()
        candle_res = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity=300", headers=headers, timeout=5).json()
        return book_res, candle_res
    except: return None, None

# --- 4. MATH ENGINE ---
def process_data(book_res, candle_res):
    if not book_res or not candle_res: return None
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values("time").reset_index(drop=True)
    
    # Stitch Live Price
    last_idx = df.index[-1]
    df.at[last_idx, 'close'] = price
    if price > df.at[last_idx, 'high']: df.at[last_idx, 'high'] = price
    if price < df.at[last_idx, 'low']: df.at[last_idx, 'low'] = price
    
    # Indicators
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi_series = 100 - (100 / (1 + rs))
    
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    vwap_series = df['pv'].cumsum() / df['vol'].cumsum()
    
    b_vol = sum([float(x[1]) for x in bids])
    a_vol = sum([float(x[1]) for x in asks])
    ofi = (b_vol - a_vol) / (b_vol + a_vol) if (b_vol+a_vol) > 0 else 0
    
    return {
        "price": price, "bids": bids, "asks": asks, "df": df,
        "rsi": rsi_series.iloc[-1], "rsi_series": rsi_series,
        "vwap": vwap_series.iloc[-1], "vwap_series": vwap_series,
        "ofi": ofi
    }

def get_walls(orders, price):
    walls = []
    threshold = 50000 if price > 1000 else 10000 
    for order in orders:
        p = float(order[0])
        s = float(order[1])
        val = p * s
        if val > threshold: walls.append((p, val))
    return walls[:3]

def run_monte_carlo(current_price, volatility=0.05, simulations=5000):
    daily_return = np.random.normal(0, volatility, simulations)
    future_prices = current_price * (1 + daily_return)
    future_prices = np.sort(future_prices)
    return np.percentile(future_prices, 5)

# --- 5. CHART ENGINE ---
def plot_chart(data, stop_loss, buy_walls, sell_walls):
    df = data['df']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=data['vwap_series'], mode='lines', name='VWAP', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="#d500f9", row=1, col=1)
    for p, v in buy_walls: fig.add_hline(y=p, line_color="rgba(0, 255, 0, 0.3)", row=1, col=1)
    for p, v in sell_walls: fig.add_hline(y=p, line_color="rgba(255, 0, 0, 0.3)", row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=data['rsi_series'], mode='lines', name='RSI', line=dict(color='#7e57c2', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", uirevision='TheTruth')
    return fig

# --- 6. APP LAYOUT ---
st.sidebar.header("⚙️ Sniper Scope")
sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()
MAP = {"GOLD": "PAXG-USD", "XAUUSD": "PAXG-USD", "BITCOIN": "BTC-USD"}
symbol = MAP.get(sym_input, sym_input)
if "-" not in symbol and len(symbol)>3: symbol = f"{symbol[:-3]}-{symbol[-3:]}"
st.sidebar.markdown("---")
trade_size = st.sidebar.number_input("Trade Size ($)", value=90.0, step=10.0)

# DEBUG BUTTON: Force a log entry to test the table
if st.sidebar.button("🛠️ Test Log"):
    log_trade(symbol, 0, 0, "TEST", "$0.00")
    st.toast("Test Row Added!")

st.title(f"🦅 {symbol} Command Center")
tab1, tab2 = st.tabs(["🚀 Dashboard", "📈 Pro Chart"])

@st.fragment(run_every=1)
def render_dashboard():
    book, candle = fetch_data(symbol)
    data = process_data(book, candle)
    if not data:
        st.warning("Loading Data...")
        return

    price = data['price']
    stop_loss = run_monte_carlo(price)
    buy_walls = get_walls(data['bids'], price)
    sell_walls = get_walls(data['asks'], price)
    dollar_risk = trade_size * ((price - stop_loss) / price)
    
    signal = "WAIT"
    score = 0
    if data['ofi'] > 0.15: score += 1
    elif data['ofi'] < -0.15: score -= 1
    if 40 < data['rsi'] < 70: 
        if score > 0: score += 1
        elif score < 0: score -= 1
    if price > data['vwap']: 
        if score > 0: score += 1
    else:
        if score < 0: score -= 1
        
    if score >= 3: signal = "PERFECT BUY 🟢"
    elif score <= -3: signal = "PERFECT SELL 🔴"

    if "PERFECT" in signal and signal != st.session_state.get('last_sig', ''):
        log_trade(symbol, price, data['ofi'], signal, f"${stop_loss:.2f}")
        st.session_state.last_sig = signal
        st.toast(f"Logged {signal}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Current Price", f"${price:,.2f}")
    c2.metric("OFI Pressure", f"{data['ofi']:.3f}")
    c3.metric("RSI Momentum", f"{data['rsi']:.1f}")
    
    st.divider()
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader(f"Signal: {signal}")
        st.caption(f"Confluence: {abs(score)}/3 Checks Passed")
    with sc2:
        risk_color = "green" if dollar_risk < (trade_size * 0.02) else "red"
        st.markdown(f"**Risk:** <span style='color:{risk_color}; font-size:20px'>-${dollar_risk:.2f}</span>", unsafe_allow_html=True)
        st.write(f"Stop Loss: ${stop_loss:,.2f}")

    st.divider()
    wc1, wc2 = st.columns(2)
    with wc1:
        st.write("🛡️ **Buy Walls**")
        for p, v in buy_walls: st.success(f"${v/1000:.0f}k @ {p:.2f}")
    with wc2:
        st.write("⚔️ **Sell Walls**")
        for p, v in sell_walls: st.error(f"${v/1000:.0f}k @ {p:.2f}")

    # --- THE TRADE JOURNAL IS BACK ---
    st.divider()
    st.subheader("📝 Trade Journal")
    try:
        if os.path.exists(LOG_FILE):
            df_log = pd.read_csv(LOG_FILE)
            # Sort newest first
            df_log = df_log.sort_values(by=df_log.columns[0], ascending=False)
            st.dataframe(df_log, use_container_width=True, height=200)
        else:
            st.info("No trades logged yet.")
    except:
        st.error("Error reading log file.")

@st.fragment(run_every=1)
def render_chart():
    book, candle = fetch_data(symbol)
    data = process_data(book, candle)
    if not data: return
    stop_loss = run_monte_carlo(data['price'])
    buy_walls = get_walls(data['bids'], data['price'])
    sell_walls = get_walls(data['asks'], data['price'])
    fig = plot_chart(data, stop_loss, buy_walls, sell_walls)
    st.plotly_chart(fig, use_container_width=True, key="live_chart_widget")

with tab1:
    render_dashboard()

with tab2:
    render_chart()
