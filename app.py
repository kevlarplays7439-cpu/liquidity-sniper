import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Liquidity Sniper Pro", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOGGING ENGINE (The Black Box) ---
LOG_FILE = "sniper_logs.csv"
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Time,Symbol,Price,OFI,RSI,Signal,Risk_VaR\n")

def log_trade(symbol, price, ofi, rsi, signal, risk):
    init_log()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{ofi:.3f},{rsi:.1f},{signal},{risk}\n")

init_log()

# --- 3. DATA ENGINE (Coinbase Institutional) ---
def get_orderbook(sym):
    try:
        url = f"https://api.exchange.coinbase.com/products/{sym}/book?level=2"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        return requests.get(url, headers=headers, timeout=5).json()
    except: return None

def get_candles(sym, granularity=300):
    """
    Fetches price history. 
    Granularity 300 = 5min candles (for RSI/VWAP)
    Granularity 86400 = 1day candles (for Volatility)
    """
    try:
        url = f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity={granularity}"
        headers = {"User-Agent": "LiquidityLens/1.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        if not res: return pd.DataFrame()
        # Coinbase returns [time, low, high, open, close, volume]
        df = pd.DataFrame(res, columns=["time", "low", "high", "open", "close", "vol"])
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except: return pd.DataFrame()

# --- 4. MATH ENGINE ---
def calculate_indicators(df):
    if df.empty: return 50, 0
    # RSI (14 Period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    vwap = df['pv'].cumsum() / df['vol'].cumsum()
    return rsi.iloc[-1], vwap.iloc[-1]

def get_real_volatility(symbol):
    """Calculates REAL Daily Volatility from last 30 days"""
    df = get_candles(symbol, granularity=86400) # Daily candles
    if df.empty: return 0.05
    returns = df['close'].pct_change().dropna()
    return returns.std()

def run_monte_carlo(current_price, volatility, simulations=1000):
    """Simulates 1,000 futures to find max downside risk"""
    future_prices = []
    for x in range(simulations):
        daily_return = np.random.normal(0, volatility)
        simulated_price = current_price * (1 + daily_return)
        future_prices.append(simulated_price)
    future_prices = np.sort(future_prices)
    var_95 = np.percentile(future_prices, 5) # 95% Confidence Level
    return var_95

def calculate_ofi(bids, asks):
    if not bids: return 0
    b_vol = sum([float(x[1]) for x in bids])
    a_vol = sum([float(x[1]) for x in asks])
    return (b_vol - a_vol) / (b_vol + a_vol)

# --- 5. UI & LOGIC ---
st.sidebar.header("⚙️ Sniper Scope")
sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()

# Symbol Mapping
MAP = {"GOLD": "PAXG-USD", "XAUUSD": "PAXG-USD", "BITCOIN": "BTC-USD", "SOL": "SOL-USD"}
symbol = MAP.get(sym_input, sym_input)
if "-" not in symbol and len(symbol)>3: symbol = f"{symbol[:-3]}-{symbol[-3:]}"

st.title(f"🦅 Liquidity Sniper: {symbol}")

# Get Data
book_data = get_orderbook(symbol)
candle_data = get_candles(symbol, 300)

if not book_data or candle_data.empty:
    st.error(f"Waiting for data on {symbol}...")
    time.sleep(1)
    st.rerun()

bids = book_data['bids']
asks = book_data['asks']
price = float(bids[0][0])
ofi = calculate_ofi(bids, asks)
rsi, vwap = calculate_indicators(candle_data)

# --- CONFLUENCE CHECK ---
signal = "WAIT"
score = 0
reasons = []

# Check 1: Order Flow
if ofi > 0.15: score += 1; reasons.append("Aggressive Buying")
elif ofi < -0.15: score -= 1; reasons.append("Aggressive Selling")

# Check 2: RSI
if 40 < rsi < 70: 
    if score > 0: score += 1
    elif score < 0: score -= 1
else: reasons.append(f"RSI Risky ({rsi:.0f})")

# Check 3: VWAP
if price > vwap: 
    if score > 0: score += 1
    reasons.append("Uptrend (Above VWAP)")
else: 
    if score < 0: score -= 1
    reasons.append("Downtrend (Below VWAP)")

# Final Verdict
if score >= 3: signal = "PERFECT BUY 🟢"
elif score <= -3: signal = "PERFECT SELL 🔴"
elif score > 0: signal = "WEAK BUY 🟡"
elif score < 0: signal = "WEAK SELL 🟠"

# --- VISUALS ---
c1, c2, c3 = st.columns(3)
c1.metric("Price", f"${price:,.2f}")
c2.metric("OFI Pressure", f"{ofi:.3f}")
c3.metric("RSI (14)", f"{rsi:.1f}")

st.divider()

# Split Layout: Signal Left, Risk Right
sc1, sc2 = st.columns([1.5, 1])

with sc1:
    st.subheader(f"🎯 Signal: {signal}")
    st.write(f"**Confluence Score:** {abs(score)}/3 Checks")
    for r in reasons: st.caption(f"• {r}")

with sc2:
    st.subheader("🎲 Risk Analysis")
    if st.button("Run Simulation (24h)"):
        with st.spinner("Simulating 1,000 Futures..."):
            vol = get_real_volatility(symbol)
            var_95 = run_monte_carlo(price, vol)
            downside = price - var_95
            
            st.error(f"⚠️ Max Risk (VaR 95%): -${downside:,.2f}")
            st.caption(f"Based on {vol*100:.2f}% Daily Volatility")
            
            # If Signal is Perfect, Auto-Log the trade
            if "PERFECT" in signal:
                log_trade(symbol, price, ofi, rsi, signal, f"-${downside:.2f}")
                st.toast("✅ Trade & Risk Metrics Logged!")

# Log Viewer
st.divider()
st.caption("📝 Trade Log")
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        st.dataframe(df.tail(3), use_container_width=True)

# Auto-Refresh
time.sleep(1)
st.rerun()
