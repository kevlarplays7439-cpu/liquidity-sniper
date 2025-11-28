import streamlit as st
import requests
import time
import pandas as pd

# 1. Page Config
st.set_page_config(page_title="Liquidity Sniper", page_icon="🦅", layout="wide")

# 2. Styles
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. The "Ask Binance" Function (HTTP Polling)
def get_binance_data():
    try:
        # We use the US API because Streamlit servers are in the US
        url = "https://api.binance.us/api/v3/depth?symbol=BTCUSDT&limit=20"
        response = requests.get(url, timeout=5)
        return response.json()
    except:
        return None

# 4. Math Helpers
def calculate_ofi(bids, asks):
    if not bids: return 0
    bid_vol = sum([float(x[1]) for x in bids])
    ask_vol = sum([float(x[1]) for x in asks])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def get_walls(orders):
    walls = []
    for p, q in orders:
        val = float(p) * float(q)
        if val > 50000: # $50k Threshold for US Exchange
            walls.append(f"${val/1000:.0f}k @ {float(p):.2f}")
    return walls[:3]

# 5. The App Layout
st.title("🦅 Liquidity Sniper (Live V4.0)")
st.caption("Institutional Order Flow Detector (US Cloud Server Edition)")

# Layout Columns
col1, col2 = st.columns([2, 1])
placeholder = st.empty()

# 6. The Loop
while True:
    data = get_binance_data()
    
    with placeholder.container():
        if not data or 'bids' not in data:
            st.warning("📡 Pinging Binance API... (If this persists, refresh page)")
            time.sleep(1)
            continue
            
        bids = data['bids']
        asks = data['asks']
        price = float(bids[0][0])
        ofi = calculate_ofi(bids, asks)
        
        # Determine Sentiment
        sentiment = "NEUTRAL ⚪"
        color = "white"
        if ofi > 0.1: 
            sentiment = "BULLISH 🟢"
            color = "#00FF00"
        elif ofi < -0.1: 
            sentiment = "BEARISH 🔴"
            color = "#FF0000"
            
        # Top Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Bitcoin Price", f"${price:,.2f}")
        m2.metric("OFI Pressure", f"{ofi:.3f}")
        m3.markdown(f"*Sentiment:* <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        
        st.divider()
        
        # Walls
        wc1, wc2 = st.columns(2)
        with wc1:
            st.write("🛡 *Buy Walls (Support)*")
            walls = get_walls(bids)
            if walls:
                for w in walls: st.success(w)
            else:
                st.info("No Walls Detected")
                
        with wc2:
            st.write("⚔ *Sell Walls (Resistance)*")
            walls = get_walls(asks)
            if walls:
                for w in walls: st.error(w)
            else:
                st.info("No Walls Detected")

    # Wait 1 second before asking again (To respect API limits)
    time.sleep(1)
