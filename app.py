import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Liquidity Sniper", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGGING ENGINE (Fixed) ---
LOG_FILE = "forward_test_logs.csv"

# Function to ensure file exists
def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Timestamp,Symbol,Price,OFI_Pressure,Sentiment,Signal,Reason\n")

def log_trade(symbol, price, ofi, sentiment, signal, reason):
    init_log_file() # Double check file exists
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp},{symbol},{price},{ofi:.4f},{sentiment},{signal},{reason}\n"
    with open(LOG_FILE, "a") as f:
        f.write(entry)

# Initialize on Startup
init_log_file()

# --- 3. HELPER FUNCTIONS ---
def clean_user_input(user_input):
    sym = user_input.replace(" ", "").upper()
    MAPPING = {
        "GOLD": "PAXG-USD", "XAUUSD": "PAXG-USD", "XAU": "PAXG-USD",
        "BITCOIN": "BTC-USD", "BTC": "BTC-USD", "ETH": "ETH-USD"
    }
    if sym in MAPPING: return MAPPING[sym]
    if not "-" in sym and len(sym) > 3: sym = f"{sym[:-3]}-{sym[-3:]}"
    return sym

def get_market_data(sym):
    try:
        url = f"https://api.exchange.coinbase.com/products/{sym}/book?level=2"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        return requests.get(url, headers=headers, timeout=5).json()
    except:
        return None

def calculate_ofi(bids, asks):
    if not bids: return 0
    bid_vol = sum([float(x[1]) for x in bids])
    ask_vol = sum([float(x[1]) for x in asks])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def get_walls(orders, price):
    walls = []
    threshold = 50000 if price > 1000 else 10000 
    for order in orders:
        p = float(order[0])
        s = float(order[1])
        val = p * s
        if val > threshold:
            walls.append(f"${val/1000:.0f}k @ {p:.2f}")
    return walls[:3]

# --- 4. SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Settings")
if "symbol" not in st.session_state:
    st.session_state.symbol = "BTC-USD"

raw_input = st.sidebar.text_input("Enter Symbol", st.session_state.symbol)
symbol = clean_user_input(raw_input)
st.session_state.symbol = raw_input 

st.sidebar.markdown("---")
st.sidebar.info(f"Tracking: {symbol}")

# DEBUG BUTTON: Force a log entry
if st.sidebar.button("🛠️ Test Log Entry"):
    log_trade(symbol, 0, 0, "TEST", "TEST_SIGNAL", "Manual Check")
    st.toast("Test Row Added!")

# --- 5. MAIN APP ---
st.title(f"🦅 Liquidity Sniper: {symbol}")

# STATE MANAGEMENT
if "last_signal" not in st.session_state:
    st.session_state.last_signal = "INIT"

# FETCH DATA
data = get_market_data(symbol)

if not data or 'bids' not in data:
    st.error(f"❌ Waiting for data for {symbol}...")
    time.sleep(1)
    st.rerun()

bids = data['bids']
asks = data['asks']
price = float(bids[0][0])
ofi = calculate_ofi(bids, asks)

# SIGNAL LOGIC
signal = "NEUTRAL"
reason = "Choppy"
if ofi > 0.15:
    signal = "BUY"
    reason = "Aggressive Buying"
elif ofi < -0.15:
    signal = "SELL"
    reason = "Aggressive Selling"

# AUTO LOGGING
if signal != st.session_state.last_signal:
    log_trade(symbol, price, ofi, signal, signal, reason)
    st.session_state.last_signal = signal

# DISPLAY METRICS
col1, col2, col3 = st.columns(3)
col1.metric("Price", f"${price:,.2f}")
col2.metric("OFI Pressure", f"{ofi:.3f}")

color = "white"
if signal == "BUY": color = "#00FF00"
if signal == "SELL": color = "#FF0000"
col3.markdown(f"**Signal:** <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)

st.divider()

# WALLS
wc1, wc2 = st.columns(2)
with wc1:
    st.write("🛡️ **Support**")
    walls = get_walls(bids, price)
    if walls:
        for w in walls: st.success(w)
    else: st.info("No Walls")
    
with wc2:
    st.write("⚔️ **Resistance**")
    walls = get_walls(asks, price)
    if walls:
        for w in walls: st.error(w)
    else: st.info("No Walls")

# --- LIVE LOG VIEWER (ALWAYS VISIBLE) ---
st.divider()
st.subheader("📝 Live Data Recorder")

# Always read file because we force-created it at the top
try:
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        # Sort by time so newest is on top
        df = df.sort_values(by="Timestamp", ascending=False)
        st.dataframe(df.head(5), use_container_width=True)
    else:
        st.info("⏳ Waiting for first signal... (Table is empty)")
except:
    st.error("Error reading log file. Resetting...")
    os.remove(LOG_FILE) # Nuke it if it's broken

# AUTO-REFRESH
time.sleep(1) 
st.rerun()
