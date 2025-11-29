import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime
import os

# --- 1. SETUP LOGGING (The Recorder) ---
LOG_FILE = "forward_test_logs.csv"

# Create file if it doesn't exist
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Symbol,Price,OFI_Pressure,Sentiment,Signal,Reason\n")

def log_trade(symbol, price, ofi, sentiment, signal, reason):
    """Saves signals to a CSV file for backtesting later."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp},{symbol},{price},{ofi:.4f},{sentiment},{signal},{reason}\n"
    with open(LOG_FILE, "a") as f:
        f.write(entry)

# --- 2. CONFIG & STYLES ---
st.set_page_config(page_title="Liquidity Sniper (Forward Test)", page_icon="🧪", layout="wide")

st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SMART SYMBOL MAPPER ---
# This converts "TradingView" names to "Coinbase" names
SYMBOL_MAP = {
    "XAUUSD": "PAXG-USD",  # Maps Gold to Paxos Gold (Digital Gold)
    "GOLD": "PAXG-USD",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD"
}

# --- 4. SIDEBAR ---
st.sidebar.header("🧪 Forward Test Lab")
user_input = st.sidebar.text_input("Enter Symbol", "XAUUSD").upper()

# Auto-Map the symbol (e.g., if user types XAUUSD, we use PAXG-USD)
if user_input in SYMBOL_MAP:
    clean_symbol = SYMBOL_MAP[user_input]
    st.sidebar.success(f"Mapped {user_input} ➡️ {clean_symbol}")
else:
    clean_symbol = user_input if "-" in user_input else f"{user_input}-USD"
    st.sidebar.info(f"Tracking: {clean_symbol}")

# --- 5. DATA ENGINE ---
def get_market_data(sym):
    try:
        # Coinbase API (Free Level 2 Data)
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
    # Dynamic Threshold: Gold/BTC needs $50k walls. Smaller coins need $10k.
    threshold = 50000 if price > 1000 else 10000 
    
    for order in orders:
        p = float(order[0])
        s = float(order[1])
        val = p * s
        if val > threshold:
            walls.append(f"${val/1000:.0f}k @ {p:.2f}")
    return walls[:3]

# --- 6. MAIN DASHBOARD ---
st.title(f"🦅 Liquidity Sniper: {user_input}")
st.caption(f"Tracking Institutional Order Flow via {clean_symbol}")

placeholder = st.empty()

# State for Signal Tracking
if "last_signal" not in st.session_state:
    st.session_state.last_signal = "NEUTRAL"

while True:
    data = get_market_data(clean_symbol)
    
    with placeholder.container():
        if not data or 'bids' not in data:
            st.warning(f"📡 Connecting to Coinbase feed for {clean_symbol}...")
            time.sleep(2)
            continue
            
        bids = data['bids']
        asks = data['asks']
        price = float(bids[0][0])
        ofi = calculate_ofi(bids, asks)
        
        # --- SIGNAL LOGIC ---
        signal = "NEUTRAL"
        reason = "Choppy"
        
        if ofi > 0.15:
            signal = "BUY"
            reason = "Aggressive Buying"
        elif ofi < -0.15:
            signal = "SELL"
            reason = "Aggressive Selling"
            
        # --- AUTO-LOGGING (The "Black Box") ---
        if signal != st.session_state.last_signal:
            log_trade(user_input, price, ofi, signal, signal, reason)
            st.session_state.last_signal = signal
            st.toast(f"🚨 New Signal: {signal} saved to CSV!")

        # --- VISUALS ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${price:,.2f}")
        col2.metric("OFI Pressure", f"{ofi:.3f}")
        
        color = "white"
        if signal == "BUY": color = "#00FF00"
        if signal == "SELL": color = "#FF0000"
        col3.markdown(f"**Signal:** <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Wall Detection
        wc1, wc2 = st.columns(2)
        buy_walls = get_walls(bids, price)
        sell_walls = get_walls(asks, price)
        
        with wc1:
            st.write("🛡️ **Support Walls**")
            if buy_walls:
                for w in buy_walls: st.success(w)
            else: st.info("No Walls")
            
        with wc2:
            st.write("⚔️ **Resistance Walls**")
            if sell_walls:
                for w in sell_walls: st.error(w)
            else: st.info("No Walls")
            
        # Live Log Preview
        st.divider()
        st.caption("📝 Live Data Recorder (Last 3 Signals)")
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            st.dataframe(df.tail(3), use_container_width=True)

    time.sleep(1)
