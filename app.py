import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
import yfinance as yf
from datetime import datetime
import os

# --- 1. CONFIG ---
st.set_page_config(page_title="Omni-Sniper Pro", page_icon="🦅", layout="wide")
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
            f.write("Time,Symbol,Price,Signal,Timeframe,StopLoss\n")

def log_trade(symbol, price, signal, tf, sl):
    init_log()
    ts = datetime.now().strftime("%H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{ts},{symbol},{price},{signal},{tf},{sl}\n")

init_log()

# --- 3. DATA ENGINE (DYNAMIC TIMEFRAME) ---
@st.cache_data(ttl=60)
def get_data(symbol, granularity):
    try:
        # Coinbase Granularity: 60, 300, 900, 3600, 21600, 86400
        headers = {"User-Agent": "FractalSearch/1.0"}
        
        # 1. Candles (History)
        candle_url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity={granularity}"
        candle_res = requests.get(candle_url, headers=headers, timeout=2).json()
        
        # 2. Orderbook (Live Walls)
        book_url = f"https://api.exchange.coinbase.com/products/{symbol}/book?level=2"
        book_res = requests.get(book_url, headers=headers, timeout=2).json()
        
        return book_res, candle_res
    except: return None, None

@st.cache_data(ttl=3600)
def get_long_term_seasonality(symbol):
    try:
        # We use Yahoo for seasonality because we need LOTS of data (1 year)
        # Coinbase only gives 300 candles max
        yf_sym = symbol if "USD" in symbol else f"{symbol}-USD"
        if "BTC" in symbol: yf_sym = "BTC-USD"
        
        df = yf.download(yf_sym, period="1y", interval="1h", progress=False)
        if df.empty: return None
        
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        # Fix column names
        df = df.rename(columns={'Date': 'time', 'Datetime': 'time', 'Close': 'close'})
        if 'time' not in df.columns: df['time'] = df.index
        
        df['time'] = pd.to_datetime(df['time'], utc=True)
        return df[['time', 'close']]
    except: return None

# --- 4. MATH ENGINE ---
def process_data(book_res, candle_res):
    if not book_res or not candle_res: return None
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.sort_values("time").reset_index(drop=True)
    
    # Stitch Live Price
    last_idx = df.index[-1]
    df.at[last_idx, 'close'] = price
    
    return df, price, bids, asks

def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0: return series
    return (series - min_val) / (max_val - min_val)

def find_patterns(df, lookback=30):
    prices = df['close'].values
    if len(prices) < lookback + 20: return [], prices
    
    current_pattern = prices[-lookback:]
    norm_target = normalize(current_pattern)
    matches = []
    
    # Scan last 300 candles only (Fast mode for live dashboard)
    for i in range(len(prices) - lookback - 20):
        candidate = prices[i : i + lookback]
        if len(candidate) == lookback:
            corr, _ = pearsonr(norm_target, normalize(candidate))
            if corr > 0.80:
                future = prices[i+lookback : i+lookback+10]
                pct = ((future[-1] - prices[i+lookback-1]) / prices[i+lookback-1]) * 100
                matches.append({
                    "date": df['time'].iloc[i], "corr": corr, 
                    "pattern": candidate, "future": future, "pct": pct
                })
    
    return sorted(matches, key=lambda x: x['corr'], reverse=True)[:3], current_pattern

def calculate_simple_seasonality(df):
    """Simple Bar Charts instead of Complex Heatmap"""
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day_name()
    df['return'] = df['close'].pct_change()
    
    # Hour Stats
    hour_stats = df.groupby('hour')['return'].mean() * 100
    
    # Day Stats (Ordered)
    day_stats = df.groupby('day')['return'].mean() * 100
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = day_stats.reindex(days)
    
    return hour_stats, day_stats

def run_risk(current_price, df_hist, timeframe_mins):
    returns = df_hist['close'].pct_change().dropna()
    vol_per_candle = returns.std()
    
    # Project risk based on timeframe
    # If 1m candles -> Project 60 mins ahead
    # If 1d candles -> Project 7 days ahead
    projection_candles = 60 if timeframe_mins < 60 else 10
    
    sim_vol = vol_per_candle * np.sqrt(projection_candles)
    sims = np.random.normal(0, sim_vol, 2000)
    futures = current_price * (1 + sims)
    
    var_95 = np.percentile(futures, 5)
    upside_95 = np.percentile(futures, 95)
    return var_95, upside_95, vol_per_candle

def get_walls(bids, asks, price):
    walls_b = []
    walls_a = []
    threshold = 50000 
    for p, s, _ in bids:
        if float(p)*float(s) > threshold: walls_b.append((float(p), float(p)*float(s)))
    for p, s, _ in asks:
        if float(p)*float(s) > threshold: walls_a.append((float(p), float(p)*float(s)))
    return walls_b[:3], walls_a[:3]

# --- 5. CHARTS ---
def plot_fractal_chart(current, match, buy_walls, sell_walls):
    fig = go.Figure()
    
    # Normalize for overlay
    scaler = current[0] / match['pattern'][0]
    hist_scaled = match['pattern'] * scaler
    fut_scaled = match['future'] * scaler
    
    x_curr = list(range(len(current)))
    x_fut = list(range(len(current), len(current)+len(fut_scaled)))
    
    fig.add_trace(go.Scatter(x=x_curr, y=current, mode='lines', name='Now', line=dict(color='#00FF00', width=3)))
    fig.add_trace(go.Scatter(x=x_curr, y=hist_scaled, mode='lines', name='History', line=dict(color='gray', dash='dot')))
    
    col = "#00c853" if match['pct'] > 0 else "#d50000"
    fig.add_trace(go.Scatter(x=x_fut, y=fut_scaled, mode='lines', name='Forecast', line=dict(color=col, width=3)))
    
    # Walls
    y_min, y_max = min(current), max(current)
    span = y_max - y_min
    for p, v in buy_walls:
        if y_min-span < p < y_max+span: fig.add_hline(y=p, line_color="green", opacity=0.5, annotation_text="Buy Wall")
    for p, v in sell_walls:
        if y_min-span < p < y_max+span: fig.add_hline(y=p, line_color="red", opacity=0.5, annotation_text="Sell Wall")

    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=0,r=0,t=10,b=0))
    return fig

def plot_bar(series, title, color_pos="#00c853", color_neg="#d50000"):
    colors = [color_pos if v > 0 else color_neg for v in series.values]
    fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=colors))
    fig.update_layout(template="plotly_dark", title=title, height=250, margin=dict(l=0,r=0,t=30,b=0))
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("🦅 Omni-Sniper")
sym = st.sidebar.text_input("Symbol", "BTC").upper()
if "-" not in sym: sym += "-USD"

# *** NEW: TIMEFRAME SELECTOR ***
tf_label = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minutes", "15 Minutes", "1 Hour", "6 Hours", "1 Day"])
# Map to Coinbase Granularity (Seconds)
tf_map = {
    "1 Minute": 60, "5 Minutes": 300, "15 Minutes": 900, 
    "1 Hour": 3600, "6 Hours": 21600, "1 Day": 86400
}
granularity = tf_map[tf_label]

st.title(f"🔍 {sym} Intelligence ({tf_label})")

if st.button("🚀 Analyze Market"):
    # 1. Fetch
    with st.spinner("Downloading Data..."):
        book_res, candle_res = get_data(sym, granularity)
        df_season = get_long_term_seasonality(sym) # Always 1H for seasonality
        
    if book_res and candle_res:
        # 2. Process
        df, price, bids, asks = process_data(book_res, candle_res)
        matches, current = find_patterns(df)
        buy_walls, sell_walls = get_walls(bids, asks, price)
        var_95, upside, vol = run_risk(price, df, granularity/60) # Pass minutes
        
        # 3. Metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${price:,.2f}")
        c2.metric("Volatility", f"{vol*100:.2f}%")
        
        risk_label = "Low" if vol < 0.005 else "High"
        c3.metric("Risk Level", risk_label)
        
        if matches:
            avg_move = np.mean([m['pct'] for m in matches])
            c4.metric("Fractal Forecast", f"{avg_move:+.2f}%", delta_color="normal" if avg_move > 0 else "inverse")
        else:
            c4.metric("Fractal Forecast", "No Match")

        st.divider()
        
        # TABS
        tab1, tab2, tab3 = st.tabs(["🔮 Pattern Match", "📅 Simple Seasonality", "🎲 Risk"])
        
        with tab1:
            if matches:
                st.caption(f"Top Match Date: {matches[0]['date'].strftime('%Y-%m-%d')}")
                st.plotly_chart(plot_fractal_chart(current, matches[0], buy_walls, sell_walls), use_container_width=True)
            else:
                st.warning("No clear history matches found for this timeframe.")
                
        with tab2:
            if df_season is not None:
                h_stats, d_stats = calculate_simple_seasonality(df_season)
                
                c_day, c_hour = st.columns(2)
                with c_day:
                    st.plotly_chart(plot_bar(d_stats, "Best Day of Week"), use_container_width=True)
                with c_hour:
                    st.plotly_chart(plot_bar(h_stats, "Best Hour (UTC)"), use_container_width=True)
                    
                # Plain English Summary
                best_day = d_stats.idxmax()
                best_hour = h_stats.idxmax()
                st.info(f"✅ **Statistical Edge:** Historically, {sym} performs best on **{best_day}s** around **{best_hour}:00 UTC**.")
            else:
                st.error("Could not fetch seasonality data.")
                
        with tab3:
            st.subheader(f"Statistical Range ({tf_label} View)")
            st.write(f"Based on current volatility, here is the 95% confidence range:")
            
            r1, r2 = st.columns(2)
            with r1:
                st.error(f"🛑 Stop Loss: ${var_95:,.2f}")
            with r2:
                st.success(f"🚀 Take Profit: ${upside:,.2f}")

    else:
        st.error("API Error. Try a different symbol or timeframe.")
