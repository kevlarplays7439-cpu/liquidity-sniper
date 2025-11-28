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

# --- SIDEBAR (USER INPUT) ---
st.sidebar.header("⚙ Configuration")
# Default is BTC-USD. User can type anything.
user_input = st.sidebar.text_input("Enter Symbol (Coinbase)", "BTC-USD")
symbol = user_input.upper() # Auto-convert 'btc-usd' to 'BTC-USD'

st.sidebar.caption(f"Tracking: {symbol}")
st.sidebar.markdown("---")
st.sidebar.write("✅ *Active Connection*")
st.sidebar.write("📡 *Source:* Coinbase Prime")

# 3. The "Ask Coinbase" Function (Dynamic)
def get_market_data(symbol):
    try:
        # We inject the {symbol} variable into the URL
        url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        response = requests.get(url, headers=headers, timeout=5)
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
    for order in orders:
        price = float(order[0])
        size = float(order[1])
        val = price * size
        # Dynamic Threshold: If price < $1000 (like SOL), use $10k wall. Else $50k.
        threshold = 10000 if price < 1000 else 50000
        
        if val > threshold: 
            walls.append(f"${val/1000:.0f}k @ {price:.2f}")
    return walls[:3]

# 5. App Layout
st.title(f"🦅 Liquidity Sniper: {symbol}")

col1, col2 = st.columns([2, 1])
placeholder = st.empty()

# 6. The Loop
while True:
    # Pass the user's symbol to the function
    data = get_market_data(symbol)
    
    with placeholder.container():
        # Error Handling: If user types "GARBAGE", Coinbase returns a message, not bids
        if not data or 'bids' not in data:
            st.error(f"❌ Could not find symbol '{symbol}'. Try 'ETH-USD' or 'SOL-USD'.")
            time.sleep(2)
            continue
            
        bids = data['bids']
        asks = data['asks']
        price = float(bids[0][0])
        ofi = calculate_ofi(bids, asks)
        
        sentiment = "NEUTRAL ⚪"
        color = "white"
        if ofi > 0.1: 
            sentiment = "BULLISH 🟢"
            color = "#00FF00"
        elif ofi < -0.1: 
            sentiment = "BEARISH 🔴"
            color = "#FF0000"
            
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${price:,.2f}")
        m2.metric("OFI Pressure", f"{ofi:.3f}")
        m3.markdown(f"*Sentiment:* <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        
        st.divider()
        
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

    time.sleep(1)
