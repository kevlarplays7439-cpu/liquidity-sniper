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
st.set_page_config(page_title="Liquidity Sniper Scalper", page_icon="🦅", layout="wide")
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
def get_data_safe(sym):
    try:
        headers = {"User-Agent": "LiquidityLens/1.0"}
        # Timeout 2s for speed
        book = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/book?level=2", headers=headers, timeout=2).json()
        candles = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity=60", headers=headers, timeout=2).json() # 1 MINUTE CANDLES
        return book, candles
    except: return None, None

# --- 4. MATH ENGINE ---
def process_data(book_res, candle_res):
    if not book_res or not candle_res: return None
    try:
        bids = book_res['bids']
        asks = book_res['asks']
        price = float(bids[0][0])
        
        # 1-Minute Candles for Scalping
        df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values("time").reset_index(drop=True)
        
        # Stitch
        last_idx = df.index[-1]
        df.at[last_idx, 'close'] = price
        if price > df.at[last_idx, 'high']: df.at[last_idx, 'high'] = price
        if price < df.at[last_idx, 'low']: df.at[last_idx, 'low'] = price
        
        # Fast RSI (7 Period instead of 14 for Scalping)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(7).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(7).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        
        # VWAP
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['pv'] = df['tp'] * df['vol']
        vwap_series = df['pv'].cumsum() / df['vol'].cumsum()
        
        b_vol = sum([float(x[1]) for x in bids])
        a_vol = sum([float(x[1]) for x in asks])
        ofi = (b_vol - a_vol) / (b_vol + a_vol) if (b_vol+a_vol) > 0 else 0
        
        # SCALPER VOLATILITY (Based on 1-min candles)
        returns = df['close'].pct_change().dropna()
        vol_1min = returns.std()
        
        return {
            "price": price, "bids": bids, "asks": asks, "df": df,
            "rsi": rsi_series.iloc[-1], "rsi_series": rsi_series,
            "vwap": vwap_series.iloc[-1], "vwap_series": vwap_series,
            "ofi": ofi, "vol_1min": vol_1min
        }
    except: return None

def get_walls(orders, price):
    walls = []
    # Lower threshold for scalping visibility ($10k walls)
    threshold = 10000 
    for order in orders:
        p = float(order[0])
        s = float(order[1])
        val = p * s
        if val > threshold: walls.append((p, val))
    return walls[:3]

def run_scalper_monte_carlo(current_price, vol_1min, simulations=2000):
    # Simulate 60 minutes into the future (Scalp Horizon)
    # Volatility scales with square root of time
    vol_hour = vol_1min * np.sqrt(60)
    
    daily_return = np.random.normal(0, vol_hour, simulations)
    future_prices = current_price * (1 + daily_return)
    future_prices = np.sort(future_prices)
    
    # 5% Worst Case
    return np.percentile(future_prices, 5)

# --- 5. CHART ---
def plot_chart(data, stop_loss, buy_walls, sell_walls):
    df = data['df']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price", increasing_line_color='#26a69a', decreasing_line_color='#ef5350'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=data['vwap_series'], mode='lines', name='VWAP', line=dict(color='#ff9800', width=1.5)), row=1, col=1)
    
    # Stop Loss Line
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="#d500f9", row=1, col=1)
    
    # Walls
    for p, v in buy_walls: fig.add_hline(y=p, line_color="rgba(0, 255, 0, 0.3)", row=1, col=1)
    for p, v in sell_walls: fig.add_hline(y=p, line_color="rgba(255, 0, 0, 0.3)", row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df['time'], y=data['rsi_series'], mode='lines', name='RSI', line=dict(color='#7e57c2', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="gray", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="gray", row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", uirevision='TheTruth')
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("⚡ Scalper Mode")
sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()
MAP = {"GOLD": "PAXG-USD", "XAUUSD": "PAXG-USD", "BITCOIN": "BTC-USD"}
symbol = MAP.get(sym_input, sym_input)
if "-" not in symbol and len(symbol)>3: symbol = f"{symbol[:-3]}-{symbol[-3:]}"
st.sidebar.markdown("---")
trade_size = st.sidebar.number_input("Trade Size ($)", value=90.0, step=10.0)

if st.sidebar.button("🛠️ Test Log"):
    log_trade(symbol, 0, 0, "TEST", "$0.00")
    st.toast("Test Row Added!")

st.title(f"🦅 {symbol} Scalper Terminal")

# MAIN LOOP
book, candle = get_data_safe(symbol)
data = process_data(book, candle)

if not data:
    st.warning(f"📡 Fetching {symbol} data...")
    time.sleep(1)
    st.rerun()

price = data['price']
# Run Scalper Monte Carlo (Tight Stop Loss)
stop_loss = run_scalper_monte_carlo(price, data['vol_1min'])
buy_walls = get_walls(data['bids'], price)
sell_walls = get_walls(data['asks'], price)
dollar_risk = trade_size * ((price - stop_loss) / price)

# --- AGGRESSIVE SIGNAL LOGIC ---
signal = "WAIT"
score = 0
reasons = []

# 1. OFI (Lower threshold for scalping)
if data['ofi'] > 0.10: score += 1; reasons.append("Buy Flow") # Was 0.15
elif data['ofi'] < -0.10: score -= 1; reasons.append("Sell Flow")

# 2. RSI (7) - Fast Momentum
if 45 < data['rsi'] < 55: reasons.append("RSI Neutral")
elif 55 <= data['rsi'] < 70: score += 1; reasons.append("Momentum Up")
elif 30 < data['rsi'] <= 45: score -= 1; reasons.append("Momentum Down")

# 3. VWAP Scalp
if price > data['vwap']: score += 1; reasons.append("Trend Up")
else: score -= 1; reasons.append("Trend Down")

# Verdict
if score >= 3: signal = "STRONG BUY 🟢"
elif score == 2: signal = "SCALP BUY 🔵" # New "Weak" Buy
elif score <= -3: signal = "STRONG SELL 🔴"
elif score == -2: signal = "SCALP SELL 🟠" # New "Weak" Sell

# Auto Log (Logs Scalp Trades too)
current_sig = f"{signal}_{price}"
if "BUY" in signal or "SELL" in signal:
    if current_sig != st.session_state.get('last_sig', ''):
        log_trade(symbol, price, data['ofi'], signal, f"${stop_loss:.2f}")
        st.session_state.last_sig = current_sig
        st.toast(f"Logged: {signal}")

# DISPLAY
tab1, tab2 = st.tabs(["⚡ Dashboard", "📈 1m Chart"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", f"${price:,.2f}")
    c2.metric("OFI (Flow)", f"{data['ofi']:.3f}")
    c3.metric("RSI (7)", f"{data['rsi']:.1f}")
    
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')} (Live)")
    st.divider()
    
    sc1, sc2 = st.columns(2)
    with sc1:
        st.subheader(f"{signal}")
        for r in reasons: st.caption(f"• {r}")
    with sc2:
        st.subheader("🎲 Risk (1 Hour)")
        risk_color = "green" if dollar_risk < (trade_size * 0.02) else "red"
        st.markdown(f"**Risk:** <span style='color:{risk_color}; font-size:20px'>-${dollar_risk:.2f}</span>", unsafe_allow_html=True)
        st.write(f"Tight SL: ${stop_loss:,.2f}")

    st.divider()
    st.caption("📝 Recent Scalp Signals")
    if os.path.exists(LOG_FILE):
        try:
            df_log = pd.read_csv(LOG_FILE).tail(5)
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        except: pass

with tab2:
    fig = plot_chart(data, stop_loss, buy_walls, sell_walls)
    st.plotly_chart(fig, use_container_width=True)

time.sleep(1)
st.rerun()
