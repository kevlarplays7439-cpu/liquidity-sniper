import streamlit as st
import requests
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from scipy.signal import argrelextrema
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
            f.write("Time,Symbol,Price,Signal,Pattern,Candle,Confluence\n")

def log_trade(symbol, price, signal, pattern, candle, confluence):
    init_log()
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{signal},{pattern},{candle},{confluence}\n")

init_log()

# --- 3. DATA ENGINE ---
def fetch_data(sym):
    try:
        headers = {"User-Agent": "LiquidityLens/1.0"}
        # Fetch 300 candles (need history for patterns)
        candles = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/candles?granularity=60", headers=headers, timeout=2).json()
        book = requests.get(f"https://api.exchange.coinbase.com/products/{sym}/book?level=2", headers=headers, timeout=2).json()
        return book, candles
    except: return None, None

# --- 4. TECHNICAL ANALYSIS ENGINE ---
def analyze_market(book_res, candle_res):
    if not book_res or not candle_res: return None
    
    # 1. PREPARE DATA
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values("time").reset_index(drop=True)
    
    # Stitch Live Price
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    last_idx = df.index[-1]
    df.at[last_idx, 'close'] = price
    if price > df.at[last_idx, 'high']: df.at[last_idx, 'high'] = price
    if price < df.at[last_idx, 'low']: df.at[last_idx, 'low'] = price

    # 2. INDICATORS
    # EMAs
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    df['vwap'] = df['pv'].cumsum() / df['vol'].cumsum()

    # Relative Volume (RVOL)
    df['vol_ma'] = df['vol'].rolling(20).mean()
    df['rvol'] = df['vol'] / df['vol_ma']

    # 3. CHART PATTERNS (Swing Detection)
    # Find local peaks (Resistance) and troughs (Support)
    n = 5 # Look 5 candles left and right
    df['min'] = df.iloc[argrelextrema(df.close.values, np.less_equal, order=n)[0]]['close']
    df['max'] = df.iloc[argrelextrema(df.close.values, np.greater_equal, order=n)[0]]['close']
    
    # Pattern Logic (Last 50 candles)
    recent_highs = df['max'].dropna().tail(5).values
    recent_lows = df['min'].dropna().tail(5).values
    
    pattern = "None"
    if len(recent_highs) >= 2:
        # Check Double Top (Two highs within 0.2% price)
        if abs(recent_highs[-1] - recent_highs[-2]) / recent_highs[-1] < 0.002:
            pattern = "Double Top (Bearish)"
    if len(recent_lows) >= 2:
        # Check Double Bottom
        if abs(recent_lows[-1] - recent_lows[-2]) / recent_lows[-1] < 0.002:
            pattern = "Double Bottom (Bullish)"

    # 4. CANDLESTICK PATTERNS (Price Action)
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    
    candle_signal = "Neutral"
    body_size = abs(curr['close'] - curr['open'])
    wick_top = curr['high'] - max(curr['close'], curr['open'])
    wick_bot = min(curr['close'], curr['open']) - curr['low']
    
    # Bullish Engulfing
    if prev['close'] < prev['open'] and curr['close'] > curr['open']: # Red then Green
        if curr['close'] > prev['open'] and curr['open'] < prev['close']:
            candle_signal = "Bullish Engulfing"
            
    # Bearish Engulfing
    elif prev['close'] > prev['open'] and curr['close'] < curr['open']: # Green then Red
        if curr['close'] < prev['open'] and curr['open'] > prev['close']:
            candle_signal = "Bearish Engulfing"
            
    # Hammer (Small body, long lower wick)
    elif wick_bot > (body_size * 2) and wick_top < body_size:
        candle_signal = "Hammer (Reversal)"

    # 5. OFI Calculation
    b_vol = sum([float(x[1]) for x in book_res['bids']])
    a_vol = sum([float(x[1]) for x in book_res['asks']])
    ofi = (b_vol - a_vol) / (b_vol + a_vol) if (b_vol+a_vol) > 0 else 0

    return {
        "price": price, "df": df, "ofi": ofi,
        "pattern": pattern, "candle": candle_signal,
        "support": recent_lows[-1] if len(recent_lows) > 0 else price * 0.9,
        "resistance": recent_highs[-1] if len(recent_highs) > 0 else price * 1.1
    }

# --- 5. CHART ENGINE ---
def plot_chart(data, buy_walls, sell_walls):
    df = data['df']
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    
    # Candles
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="Price"), row=1, col=1)
    
    # EMAs
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_9'], line=dict(color='yellow', width=1), name="EMA 9"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['time'], y=df['ema_50'], line=dict(color='purple', width=1), name="EMA 50"), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], line=dict(color='orange', width=2), name="VWAP"), row=1, col=1)
    
    # Support/Resistance Lines
    fig.add_hline(y=data['support'], line_dash="dash", line_color="green", annotation_text="Support")
    fig.add_hline(y=data['resistance'], line_dash="dash", line_color="red", annotation_text="Resistance")

    # RSI
    fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], line=dict(color='#7e57c2', width=1.5), name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="gray", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="gray", dash="dot"), row=2, col=1)
    
    fig.update_layout(template="plotly_dark", height=600, margin=dict(l=0, r=0, t=10, b=0), xaxis_rangeslider_visible=False, uirevision='TheTruth')
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("⚙️ Sniper Scope")
sym = st.sidebar.text_input("Symbol", "BTC-USD").upper()
if "-" not in sym and len(sym)>3: sym = f"{sym[:-3]}-{sym[-3:]}"

if st.sidebar.button("🛠️ Test Log"):
    log_trade(sym, 0, "TEST", "None", "None", "Manual")
    st.toast("Test Row Added!")

st.title(f"🦅 {sym} Technical Confluence")

@st.fragment(run_every=1)
def main_loop():
    book, candle = fetch_data(sym)
    data = analyze_market(book, candle)
    
    if not data:
        st.warning("📡 Analyzing Market Structure...")
        return

    # --- CONFLUENCE SCORE ---
    score = 0
    reasons = []
    
    # 1. Trend (Price > EMA 50)
    if data['price'] > data['df']['ema_50'].iloc[-1]: score += 1; reasons.append("Uptrend (EMA 50)")
    else: score -= 1; reasons.append("Downtrend (EMA 50)")
    
    # 2. Momentum (RSI)
    rsi = data['df']['rsi'].iloc[-1]
    if 45 < rsi < 65: reasons.append("RSI Neutral")
    elif rsi >= 65: score += 1; reasons.append("Momentum High")
    elif rsi <= 35: score -= 1; reasons.append("Momentum Low")
    
    # 3. Order Flow (OFI)
    if data['ofi'] > 0.15: score += 1; reasons.append("Whale Buying")
    elif data['ofi'] < -0.15: score -= 1; reasons.append("Whale Selling")
    
    # 4. Patterns
    if "Bullish" in data['pattern'] or "Bullish" in data['candle']: score += 2; reasons.append(f"Pattern: {data['pattern'] or data['candle']}")
    if "Bearish" in data['pattern'] or "Bearish" in data['candle']: score -= 2; reasons.append(f"Pattern: {data['pattern'] or data['candle']}")

    # Verdict
    signal = "WAIT"
    if score >= 3: signal = "STRONG BUY 🟢"
    elif score <= -3: signal = "STRONG SELL 🔴"
    
    # Auto Log
    if "STRONG" in signal and signal != st.session_state.get('last_sig', ''):
        log_trade(sym, data['price'], signal, data['pattern'], data['candle'], f"Score: {score}")
        st.session_state.last_sig = signal
        st.toast(f"Signal: {signal}")

    # Display
    tab1, tab2 = st.tabs(["📊 Analysis", "📈 Chart"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${data['price']:.2f}")
        c2.metric("Pattern", data['pattern'] if data['pattern'] != "None" else "-")
        c3.metric("Candle", data['candle'] if data['candle'] != "Neutral" else "-")
        c4.metric("RVOL", f"{data['df']['rvol'].iloc[-1]:.1f}x")
        
        st.divider()
        st.subheader(f"{signal} (Score: {score})")
        st.write("Confluence Factors:")
        for r in reasons: st.caption(f"✅ {r}")
        
        st.divider()
        st.caption("📝 Trade Journal")
        if os.path.exists(LOG_FILE):
            st.dataframe(pd.read_csv(LOG_FILE).tail(3), use_container_width=True, hide_index=True)

    with tab2:
        st.plotly_chart(plot_chart(data, [], []), use_container_width=True)

main_loop()
