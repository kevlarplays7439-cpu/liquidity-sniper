import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
import yfinance as yf

# --- CONFIG ---
st.set_page_config(page_title="Omni-Sniper Pro", page_icon="🦅", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.1rem; }
    .block-container { padding-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINES ---
@st.cache_data(ttl=60)
def get_live_book(symbol):
    try:
        if "-" not in symbol and "USD" not in symbol: symbol = f"{symbol}-USD"
        headers = {"User-Agent": "FractalSearch/1.0"}
        book = requests.get(f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2", headers=headers, timeout=2).json()
        return book
    except: return None

@st.cache_data(ttl=3600)
def get_long_term_data(symbol):
    try:
        yf_sym = f"{symbol}-USD" if "USD" not in symbol else symbol
        if "BTC" in symbol: yf_sym = "BTC-USD"
        
        df = yf.download(yf_sym, period="1y", interval="1h", progress=False)
        if df.empty: return None
        
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        mapper = {'Date': 'time', 'Datetime': 'time', 'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'vol'}
        df = df.rename(columns=mapper)
        
        if 'time' not in df.columns and 'index' in df.columns: df['time'] = df['index']
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df[['time', 'close', 'vol']]
    except: return None

# --- 2. MATH ENGINES ---
def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0: return series
    return (series - min_val) / (max_val - min_val)

def find_patterns(df, lookback=30):
    if len(df) < lookback + 20: return [], df['close'].tail(lookback).values
    
    prices = df['close'].values
    dates = df['time'].values
    current_pattern = prices[-lookback:]
    norm_target = normalize(current_pattern)
    
    matches = []
    # Limit scan to recent history for speed (last 4000 candles)
    scan_len = min(len(prices) - lookback - 10, 4000) 
    start_idx = len(prices) - scan_len - lookback - 10
    
    # Check if start_idx is valid
    if start_idx < 0: start_idx = 0

    prog_bar = st.progress(0)
    step = max(1, scan_len // 50)
    
    for i in range(start_idx, len(prices) - lookback - 10):
        if i % step == 0: prog_bar.progress(min((i - start_idx) / scan_len, 1.0))
            
        candidate = prices[i : i + lookback]
        if len(candidate) == len(norm_target):
            # Pre-filter for speed
            if (candidate[-1] > candidate[0]) == (current_pattern[-1] > current_pattern[0]):
                corr, _ = pearsonr(norm_target, normalize(candidate))
                if corr > 0.85:
                    future = prices[i+lookback : i+lookback+12]
                    pct = ((future[-1] - prices[i+lookback-1]) / prices[i+lookback-1]) * 100
                    matches.append({
                        "date": dates[i], "corr": corr, "pattern": candidate, 
                        "future": future, "pct": pct
                    })
    
    prog_bar.empty()
    return sorted(matches, key=lambda x: x['corr'], reverse=True)[:3], current_pattern

def get_walls(book, current_price):
    if not book: return [], []
    try:
        bids = book['bids']
        asks = book['asks']
        
        buy_walls = []
        sell_walls = []
        threshold = 50000 
        
        for p, s, _ in bids:
            val = float(p) * float(s)
            if val > threshold: buy_walls.append((float(p), val))
                
        for p, s, _ in asks:
            val = float(p) * float(s)
            if val > threshold: sell_walls.append((float(p), val))
            
        return buy_walls[:3], sell_walls[:3]
    except: return [], []

# --- FIXED RISK FUNCTION (RETURNING 3 VALUES NOW) ---
def run_risk(current_price, df_hist):
    returns = df_hist['close'].pct_change().dropna()
    vol = returns.std() * np.sqrt(24) # Daily Volatility
    sims = np.random.normal(0, vol * np.sqrt(7), 5000) # 7 Day Horizon
    futures = current_price * (1 + sims)
    
    var_95 = np.percentile(futures, 5)
    upside_95 = np.percentile(futures, 95)
    
    # BUG FIX: Now returning 3 values (Risk, Upside, Volatility)
    return var_95, upside_95, vol

# --- 3. CHART ENGINE ---
def plot_combo_chart(current_pattern, match, buy_walls, sell_walls):
    fig = go.Figure()
    
    curr_start = current_pattern[0]
    hist_start = match['pattern'][0]
    scaler = curr_start / hist_start
    
    scaled_hist = match['pattern'] * scaler
    scaled_future = match['future'] * scaler
    
    x_curr = list(range(len(current_pattern)))
    x_fut = list(range(len(current_pattern), len(current_pattern) + len(scaled_future)))
    
    fig.add_trace(go.Scatter(x=x_curr, y=current_pattern, mode='lines', name='Price Now', line=dict(color='#00FF00', width=3)))
    fig.add_trace(go.Scatter(x=x_curr, y=scaled_hist, mode='lines', name=f"Fractal ({pd.to_datetime(match['date']).strftime('%Y')})", line=dict(color='gray', width=2, dash='dot')))
    
    color = "#00c853" if match['pct'] > 0 else "#d50000"
    fig.add_trace(go.Scatter(x=x_fut, y=scaled_future, mode='lines', name='Projection', line=dict(color=color, width=3)))
    
    y_min, y_max = min(current_pattern), max(current_pattern)
    range_span = y_max - y_min
    
    for p, v in buy_walls:
        if y_min - range_span < p < y_max + range_span:
            fig.add_hline(y=p, line_color="rgba(0, 255, 0, 0.5)", annotation_text=f"🐳 Buy ${v/1000:.0f}k")
    for p, v in sell_walls:
        if y_min - range_span < p < y_max + range_span:
            fig.add_hline(y=p, line_color="rgba(255, 0, 0, 0.5)", annotation_text=f"🐳 Sell ${v/1000:.0f}k", annotation_position="top right")

    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="Hours", yaxis_title="Price ($)")
    return fig

def plot_seasonality(df):
    df['hour'] = df['time'].dt.hour
    df['return'] = df['close'].pct_change()
    stats = df.groupby('hour')['return'].mean() * 100
    
    colors = ['#00c853' if v > 0 else '#d50000' for v in stats.values]
    fig = go.Figure(go.Bar(x=stats.index, y=stats.values, marker_color=colors))
    fig.update_layout(template="plotly_dark", title="Hourly Seasonality (UTC)", yaxis_title="Avg Return %", height=300)
    return fig

# --- 4. MAIN APP ---
st.sidebar.header("🦅 Omni-Sniper")
sym = st.sidebar.text_input("Symbol", "BTC").upper()
if "-" not in sym: sym += "-USD"

st.title(f"🔍 {sym} Market Intelligence")

if st.button("🚀 Analyze Market"):
    df_hist = None
    book = None
    with st.spinner("Downloading History & Orderbook..."):
        df_hist = get_long_term_data(sym)
        book = get_live_book(sym)
        
    if df_hist is not None and book is not None:
        matches, current = find_patterns(df_hist)
        buy_walls, sell_walls = get_walls(book, current[-1])
        
        # --- FIXED CALL ---
        var_95, upside, vol = run_risk(current[-1], df_hist)
        
        if matches:
            avg_move = np.mean([m['pct'] for m in matches])
            win_rate = sum(1 for m in matches if m['pct'] > 0) / len(matches) * 100
        else:
            avg_move, win_rate = 0, 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Price", f"${current[-1]:,.2f}")
        c2.metric("Hist. Win Rate", f"{win_rate:.0f}%")
        c3.metric("Projected Move", f"{avg_move:+.2f}%", delta_color="normal" if avg_move > 0 else "inverse")
        c4.metric("Risk (VaR 95)", f"${var_95:,.0f}")
        
        st.divider()
        
        tab1, tab2, tab3 = st.tabs(["🔮 Pattern + Walls", "📅 Seasonality", "🎲 Risk"])
        
        with tab1:
            st.caption("Green/Red Lines = Real-Time Liquidity Walls (Whales)")
            st.caption("Dotted Line = Historical Pattern Match")
            if matches:
                st.subheader(f"Best Match: {pd.to_datetime(matches[0]['date']).strftime('%Y-%m-%d')}")
                st.plotly_chart(plot_combo_chart(current, matches[0], buy_walls, sell_walls), use_container_width=True)
            else:
                st.warning("No patterns found.")
                
        with tab2:
            st.subheader("Best Time to Trade")
            st.plotly_chart(plot_seasonality(df_hist), use_container_width=True)
            
        with tab3:
            st.info(f"Based on {vol*100:.2f}% Daily Volatility (1 Year History):")
            st.success(f"Optimistic Target: ${upside:,.2f}")
            st.error(f"Defensive Stop Loss: ${var_95:,.2f}")

    else:
        st.error("Data Error. Yahoo/Coinbase might be blocking. Try local run.")
