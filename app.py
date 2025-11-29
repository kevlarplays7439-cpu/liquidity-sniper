import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime
import os

# --- 1. SETUP DYNAMIC LOGGING ---
LOG_FILE = "forward_test_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w") as f:
        f.write("Timestamp,Symbol,Price,OFI_Pressure,Sentiment,Signal,Reason\n")

def log_trade(symbol, price, ofi, sentiment, signal, reason):
    # NOW USES THE 'symbol' VARIABLE PASSED TO IT
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"{timestamp},{symbol},{price},{ofi:.4f},{sentiment},{signal},{reason}\n"
    with open(LOG_FILE, "a") as f:
        f.write(entry)

# --- 2. CONFIG ---
st.set_page_config(page_title="Liquidity Sniper", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. SMART CLEANER ---
def clean_user_input(user_input):
    sym = user_input.replace(" ", "").upper()
    MAPPING = {
        "GOLD": "PAXG-USD",
        "XAUUSD": "PAXG-USD",
        "XAU": "PAXG-USD",
        "BITCOIN": "BTC-USD",
        "BTC": "BTC-USD",
        "ETH": "ETH-USD"
    }
    if sym in MAPPING: return MAPPING[sym]
    if not "-" in sym and len(sym) > 3: sym = f"{sym[:-3]}-{sym[-3:]}"
    return sym

# --- 4. SIDEBAR INPUT ---
st.sidebar.header("⚙️ Settings")
# CHANGED DEFAULT BACK TO BTC-USD
raw_input = st.sidebar.text_input("Enter Symbol", "BTC-USD") 
symbol = clean_user_input(raw_input)

st.sidebar.markdown("---")
st.sidebar.info(f"Tracking: {symbol}")

# --- 5. DATA ENGINE ---
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

# --- 6. DASHBOARD ---
st.title(f"🦅 Liquidity Sniper: {symbol}")

placeholder = st.empty()
if "last_signal" not in st.session_state:
    st.session_state.last_signal = "NEUTRAL"

while True:
    data = get_market_data(symbol)
    
    with placeholder.container():
        if not data or 'bids' not in data:
            st.error(f"❌ Waiting for data for {symbol}...")
            time.sleep(2)
            continue
            
        bids = data['bids']
        asks = data['asks']
        price = float(bids[0][0])
        ofi = calculate_ofi(bids, asks)
        
        # LOGIC
        signal = "NEUTRAL"
        reason = "Choppy"
        if ofi > 0.15:
            signal = "BUY"
            reason = "Aggressive Buying"
        elif ofi < -0.15:
            signal = "SELL"
            reason = "Aggressive Selling"
            
        # LOGGING (THE FIX IS HERE)
        # We check if signal changed, then pass the CURRENT 'symbol' to the log
        if signal != st.session_state.last_signal:
            log_trade(symbol, price, ofi, signal, signal, reason) # <--- PASSING DYNAMIC SYMBOL
            st.session_state.last_signal = signal
            st.toast(f"🚨 Logged {signal} for {symbol}!")

        # VISUALS
        col1, col2, col3 = st.columns(3)
        col1.metric("Price", f"${price:,.2f}")
        col2.metric("OFI Pressure", f"{ofi:.3f}")
        
        color = "white"
        if signal == "BUY": color = "#00FF00"
        if signal == "SELL": color = "#FF0000"
        col3.markdown(f"**Signal:** <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)
        
        st.divider()
        
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

    time.sleep(1)
