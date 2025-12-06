import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime
import os

# --- 1. PREMIUM CONFIG & CSS ---
st.set_page_config(page_title="Liquidity Sniper Pro", page_icon="🦅", layout="wide")

st.markdown("""
    <style>
    /* GLOBAL DARK THEME */
    .stApp {
        background-color: #050505;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* ANIMATIONS */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 127, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 127, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 127, 0); }
    }
    .live-dot {
        height: 12px; width: 12px;
        background-color: #00ff7f;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 2s infinite;
        margin-right: 8px;
    }

    /* GLASS CARDS */
    .metric-card {
        background: rgba(20, 20, 20, 0.7);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    .metric-value { color: #fff; font-size: 28px; font-weight: 700; margin-top: 5px; }
    .metric-label { color: #888; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-sub { font-size: 12px; margin-top: 5px; font-weight: 500; }
    
    /* TABS */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #000000; padding: 5px; border-radius: 30px; border: 1px solid #333; }
    .stTabs [data-baseweb="tab"] { height: 40px; border-radius: 20px; color: #888; font-size: 13px; border: none; }
    .stTabs [aria-selected="true"] { background-color: #1f1f1f !important; color: #fff !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOGGING ENGINE ---
LOG_FILE = "sniper_logs.csv"
def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("Time,Symbol,Price,Signal,Reason,Risk\n")

def log_trade(symbol, price, signal, reason, risk):
    init_log()
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{signal},{reason},{risk}\n")

init_log()

# --- 3. DATA ENGINE ---
@st.cache_data(ttl=60)
def get_data(symbol, granularity):
    try:
        headers = {"User-Agent": "FractalSearch/1.0"}
        candle_url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        book_url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        return requests.get(book_url, headers=headers, timeout=2).json(), requests.get(candle_url, headers=headers, timeout=2).json()
    except: return None, None

@st.cache_data(ttl=3600)
def get_history(symbol):
    try:
        yf_sym = symbol if "USD" in symbol else f"{symbol}-USD"
        if "BTC" in symbol: yf_sym = "BTC-USD"
        df = yf.download(yf_sym, period="1y", interval="1h", progress=False)
        if df.empty: return None
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        df = df.rename(columns={'Date': 'time', 'Datetime': 'time', 'Close': 'close'})
        if 'time' not in df.columns: df['time'] = df.index
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df[['time', 'close']]
    except: return None

# --- 4. MATH ENGINE (FIXED) ---
def process_data(book_res, candle_res):
    if not book_res or not candle_res: return None
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    
    last_idx = df.index[-1]
    df.at[last_idx, 'close'] = price
    if price > df.at[last_idx, 'high']: df.at[last_idx, 'high'] = price
    if price < df.at[last_idx, 'low']: df.at[last_idx, 'low'] = price
    
    return df, price, bids, asks

def get_indicators(df):
    # EMA
    df['ema'] = df['close'].ewm(span=50).mean()
    
    # RSI (Simplified Logic to prevent Syntax Error)
    delta = df['close'].diff()
    
    # Separate gains and losses
    up = delta.copy()
    up[up < 0] = 0
    down = delta.copy()
    down[down > 0] = 0
    
    # Calculate averages
    gain = up.rolling(14).mean()
    loss = abs(down.rolling(14).mean())
    
    # Calculate RSI
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # VWAP
    df['tp'] = (df['high'] + df['low'] + df['close']) / 3
    df['pv'] = df['tp'] * df['vol']
    df['vwap'] = df['pv'].cumsum() / df['vol'].cumsum()
    
    return df

def get_walls(bids, asks, price):
    walls_b, walls_a = [], []
    threshold = 50000 
    for p, s, _ in bids:
        if float(p)*float(s) > threshold: walls_b.append((float(p), float(p)*float(s)))
    for p, s, _ in asks:
        if float(p)*float(s) > threshold: walls_a.append((float(p), float(p)*float(s)))
    return walls_b[:3], walls_a[:3]

def run_risk(current_price, df_hist):
    returns = df_hist['close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(24) 
    sims = np.random.normal(0, vol * np.sqrt(7), 5000)
    futures = current_price * (1 + sims)
    return np.percentile(futures, 5), np.percentile(futures, 95), vol

# --- 5. VISUALS ---
def plot_chart(df, stop_loss, buy_walls, sell_walls):
    fig = go.Figure()
    
    # Candles
    fig.add_trace(go.Candlestick(x=df['time'], open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name="Price", increasing_line_color='#00ff7f', decreasing_line_color='#ff3d3d'))
    
    # VWAP
    fig.add_trace(go.Scatter(x=df['time'], y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#ff9800', width=2)))
    
    # Stop Loss
    fig.add_hline(y=stop_loss, line_dash="dot", line_color="#d500f9", annotation_text="VaR Risk")
    
    # Walls
    for p, v in buy_walls: fig.add_hline(y=p, line_color="rgba(0, 255, 127, 0.3)")
    for p, v in sell_walls: fig.add_hline(y=p, line_color="rgba(255, 61, 61, 0.3)")

    fig.update_layout(
        template="plotly_dark", height=500, margin=dict(l=0,r=0,t=20,b=0),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#222'),
        uirevision='TheTruth'
    )
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("🦅 Omni-Sniper")
sym = st.sidebar.text_input("Symbol", "BTC").upper()
if "-" not in sym: sym += "-USD"
tf_label = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minutes", "15 Minutes", "1 Hour", "1 Day"])
tf_map = {"1 Minute": 60, "5 Minutes": 300, "15 Minutes": 900, "1 Hour": 3600, "1 Day": 86400}
granularity = tf_map[tf_label]
trade_size = st.sidebar.number_input("Trade Size ($)", value=90.0)

st.markdown(f"### <div class='live-dot'></div> {sym} LIVE INTELLIGENCE", unsafe_allow_html=True)

def render_card(label, value, sub, color="white"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color}">{value}</div>
        <div class="metric-sub" style="color: {color}">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

@st.fragment(run_every=1)
def dashboard():
    book_res, candle_res = get_data(sym, granularity)
    df_hist = get_history(sym)
    
    if book_res and candle_res and df_hist is not None:
        df, price, bids, asks = process_data(book_res, candle_res)
        df = get_indicators(df)
        buy_walls, sell_walls = get_walls(bids, asks, price)
        var_95, upside, vol = run_risk(price, df_hist)
        
        # Logic
        b_vol = sum([float(x[1]) for x in bids])
        a_vol = sum([float(x[1]) for x in asks])
        ofi_val = (b_vol - a_vol) / (b_vol + a_vol) if (b_vol+a_vol) > 0 else 0
        
        score = 0
        if ofi_val > 0.15: score += 1
        elif ofi_val < -0.15: score -= 1
        if price > df['vwap'].iloc[-1]: score += 1
        else: score -= 1
        
        signal = "WAIT"
        sig_color = "#888"
        if score >= 2: signal = "STRONG BUY"; sig_color = "#00ff7f"
        elif score <= -2: signal = "STRONG SELL"; sig_color = "#ff3d3d"
        
        risk_dol = trade_size * ((price - var_95) / price)
        risk_color = "#ff3d3d" if risk_dol > (trade_size*0.02) else "#00ff7f"

        if "STRONG" in signal and signal != st.session_state.get('last_sig', ''):
            log_trade(sym, price, signal, f"Score: {score}", f"${var_95:.2f}")
            st.session_state.last_sig = signal
            st.toast(f"Logged: {signal}")

        c1, c2, c3, c4 = st.columns(4)
        with c1: render_card("MARKET PRICE", f"${price:,.2f}", f"Vol: {vol*100:.2f}%")
        with c2: render_card("SIGNAL", signal, f"Confluence: {score}/3", sig_color)
        with c3: render_card("RISK (VaR)", f"-${risk_dol:.2f}", f"Stop: ${var_95:,.0f}", risk_color)
        with c4: render_card("OFI FLOW", f"{ofi_val:.3f}", "Institutional Pressure", "#00ff7f" if ofi_val>0 else "#ff3d3d")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📈 Command Center", "📝 Trade Journal"])
        
        with tab1:
            st.plotly_chart(plot_chart(df, var_95, buy_walls, sell_walls), use_container_width=True)
            w1, w2 = st.columns(2)
            with w1: 
                if buy_walls: st.success(f"Support Wall: ${buy_walls[0][0]:,.2f}")
            with w2:
                if sell_walls: st.error(f"Resistance Wall: ${sell_walls[0][0]:,.2f}")

        with tab2:
            if os.path.exists(LOG_FILE):
                st.dataframe(pd.read_csv(LOG_FILE).tail(5), use_container_width=True, hide_index=True)

    else: st.caption("📡 Connecting to Exchange...")

dashboard()
