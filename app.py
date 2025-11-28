import streamlit as st
import websocket
import json
import threading
import time

# 1. Config the Page
st.set_page_config(page_title="Liquidity Sniper", page_icon="🦅", layout="wide")

# 2. Add Custom Styling (Dark Mode Matrix Look)
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; border: 1px solid #333; padding: 15px; border-radius: 5px; }
    .bullish { color: #00FF00; font-weight: bold; }
    .bearish { color: #FF0000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Setup Memory (Session State)
if 'price' not in st.session_state: st.session_state.price = 0.0
if 'bids' not in st.session_state: st.session_state.bids = []
if 'asks' not in st.session_state: st.session_state.asks = []

# 4. Define the WebSocket Connection (Background Worker)
def on_message(ws, message):
    data = json.loads(message)
    # Save live data to memory
    st.session_state.price = float(data['bids'][0][0])
    st.session_state.bids = data['bids']
    st.session_state.asks = data['asks']

def start_socket():
    # We use binance.us because Cloud Servers are usually in the USA
    url = "wss://stream.binance.us:9443/ws/btcusdt@depth20@100ms"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

# Start the worker thread only once
if 'thread_started' not in st.session_state:
    t = threading.Thread(target=start_socket)
    t.daemon = True
    t.start()
    st.session_state.thread_started = True

# 5. Helper Math Functions
def calculate_ofi(bids, asks):
    if not bids: return 0
    bid_vol = sum([float(x[1]) for x in bids])
    ask_vol = sum([float(x[1]) for x in asks])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol)

def get_walls(orders):
    # Find orders worth more than $100k
    walls = []
    for p, q in orders:
        val = float(p) * float(q)
        if val > 100000: # $100k Threshold
            walls.append(f"${val/1000:.0f}k @ {float(p):.2f}")
    return walls[:3]

# 6. Build the Dashboard UI
st.title("🦅 Liquidity Sniper (Live)")
st.caption("Real-Time Institutional Order Flow Detector")

# Create a container that refreshes
placeholder = st.empty()

while True:
    with placeholder.container():
        # Get latest data
        price = st.session_state.price
        bids = st.session_state.bids
        asks = st.session_state.asks
        
        if price == 0:
            st.warning("Connecting to Live Market Feed... Please Wait...")
            time.sleep(1)
            continue
            
        ofi = calculate_ofi(bids, asks)
        
        # Top Row: Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Bitcoin Price", f"${price:,.2f}")
        
        sentiment = "NEUTRAL ⚪"
        if ofi > 0.1: sentiment = "BULLISH 🟢"
        if ofi < -0.1: sentiment = "BEARISH 🔴"
        
        c2.metric("Pressure Score", f"{ofi:.3f}")
        c3.write(f"### {sentiment}")
        
        st.markdown("---")
        
        # Bottom Row: Walls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛡 Buy Walls (Support)")
            walls = get_walls(bids)
            if walls:
                for w in walls: st.success(w)
            else:
                st.info("No Whales Detected")
                
        with col2:
            st.subheader("⚔ Sell Walls (Resistance)")
            walls = get_walls(asks)
            if walls:
                for w in walls: st.error(w)
            else:
                st.info("No Whales Detected")
        
        time.sleep(1) # Refresh every 1 second