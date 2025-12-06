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

# --- 1. CONFIG & PREMIUM CSS ---
st.set_page_config(page_title="Omni-Sniper Pro", page_icon="🦅", layout="wide")

# THE EXPENSIVE UI INJECTION
st.markdown("""
    <style>
    /* 1. Main Background & Font */
    .stApp {
        background-color: #050505;
        font-family: 'Roboto', sans-serif;
    }
    
    /* 2. Remove Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 3. Glassmorphism Cards */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #111111;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* 4. Metric Value Styling (Big & Bright) */
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        color: #ffffff !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #888888 !important;
    }
    
    /* 5. Custom Button (Neon Green) */
    .stButton > button {
        background: linear-gradient(45deg, #00c853, #009624);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 10px #00c853;
    }
    
    /* 6. Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #0e0e0e;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border-radius: 5px;
        color: white;
        font-size: 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #333 !important;
        color: #00c853 !important;
        border: 1px solid #00c853;
    }
    
    /* 7. Remove Padding */
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
def get_long_term_seasonality(symbol):
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
        df['time'] = df['time'].dt.tz_convert('Asia/Kolkata')
        return df[['time', 'close']]
    except: return None

# --- 4. MATH ENGINE ---
def process_data(book_res, candle_res):
    if not book_res or not candle_res: return None
    bids = book_res['bids']
    asks = book_res['asks']
    price = float(bids[0][0])
    
    df = pd.DataFrame(candle_res, columns=["time", "low", "high", "open", "close", "vol"])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['time'] = df['time'].dt.tz_convert('Asia/Kolkata')
    df = df.sort_values("time").reset_index(drop=True)
    
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
    
    for i in range(len(prices) - lookback - 20):
        candidate = prices[i : i + lookback]
        if len(candidate) == lookback:
            corr, _ = pearsonr(norm_target, normalize(candidate))
            if corr > 0.80:
                future = prices[i+lookback : i+lookback+10]
                pct = ((future[-1] - prices[i+lookback-1]) / prices[i+lookback-1]) * 100
                matches.append({"date": df['time'].iloc[i], "corr": corr, "pattern": candidate, "future": future, "pct": pct})
    return sorted(matches, key=lambda x: x['corr'], reverse=True)[:3], current_pattern

def calculate_simple_seasonality(df):
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day_name()
    df['return'] = df['close'].pct_change()
    hour_stats = df.groupby('hour')['return'].mean() * 100
    day_stats = df.groupby('day')['return'].mean() * 100
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_stats = day_stats.reindex(days)
    return hour_stats, day_stats

def run_risk(current_price, df_hist, timeframe_mins):
    returns = df_hist['close'].pct_change().dropna()
    vol_per_candle = returns.std()
    projection_candles = 60 if timeframe_mins < 60 else 10
    sim_vol = vol_per_candle * np.sqrt(projection_candles)
    sims = np.random.normal(0, sim_vol, 2000)
    futures = current_price * (1 + sims)
    return np.percentile(futures, 5), np.percentile(futures, 95), vol_per_candle

def get_walls(bids, asks, price):
    walls_b = []
    walls_a = []
    threshold = 50000 
    for p, s, _ in bids:
        if float(p)*float(s) > threshold: walls_b.append((float(p), float(p)*float(s)))
    for p, s, _ in asks:
        if float(p)*float(s) > threshold: walls_a.append((float(p), float(p)*float(s)))
    return walls_b[:3], walls_a[:3]

# --- 5. CHARTS (PREMIUM LOOK) ---
def plot_fractal_chart(current, match, buy_walls, sell_walls):
    fig = go.Figure()
    scaler = current[0] / match['pattern'][0]
    hist_scaled = match['pattern'] * scaler
    fut_scaled = match['future'] * scaler
    x_curr = list(range(len(current)))
    x_fut = list(range(len(current), len(current)+len(fut_scaled)))
    
    # Neon Green Current Line
    fig.add_trace(go.Scatter(x=x_curr, y=current, mode='lines', name='Now', line=dict(color='#00FF00', width=3, shape='spline')))
    # Dotted Grey History
    fig.add_trace(go.Scatter(x=x_curr, y=hist_scaled, mode='lines', name='History', line=dict(color='#555555', dash='dot')))
    # Forecast
    col = "#00c853" if match['pct'] > 0 else "#d50000"
    fig.add_trace(go.Scatter(x=x_fut, y=fut_scaled, mode='lines', name='Forecast', line=dict(color=col, width=4)))
    
    y_min, y_max = min(current), max(current)
    span = y_max - y_min
    for p, v in buy_walls:
        if y_min-span < p < y_max+span: fig.add_hline(y=p, line_color="rgba(0, 255, 0, 0.4)", annotation_text="Buy")
    for p, v in sell_walls:
        if y_min-span < p < y_max+span: fig.add_hline(y=p, line_color="rgba(255, 0, 0, 0.4)", annotation_text="Sell")

    # Clean Grid Layout
    fig.update_layout(
        template="plotly_dark", 
        height=500, 
        margin=dict(l=0,r=0,t=10,b=0),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='#222222')
    )
    return fig

def plot_bar(series, title, color_pos="#00c853", color_neg="#d50000"):
    colors = [color_pos if v > 0 else color_neg for v in series.values]
    fig = go.Figure(go.Bar(x=series.index, y=series.values, marker_color=colors))
    fig.update_layout(
        template="plotly_dark", 
        title=dict(text=title, font=dict(size=14, color="#888")),
        height=250, 
        margin=dict(l=0,r=0,t=40,b=0),
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117'
    )
    return fig

# --- 6. MAIN APP ---
st.sidebar.header("🦅 Omni-Sniper")
sym = st.sidebar.text_input("Symbol", "BTC").upper()
if "-" not in sym: sym += "-USD"

tf_label = st.sidebar.selectbox("Timeframe", ["1 Minute", "5 Minutes", "15 Minutes", "1 Hour", "6 Hours", "1 Day"])
tf_map = {"1 Minute": 60, "5 Minutes": 300, "15 Minutes": 900, "1 Hour": 3600, "6 Hours": 21600, "1 Day": 86400}
granularity = tf_map[tf_label]

st.title(f"🔍 {sym} Market Intelligence")

if st.button("🚀 Analyze Market"):
    with st.spinner("Crunching Institutional Data..."):
        book_res, candle_res = get_data(sym, granularity)
        df_season = get_long_term_seasonality(sym)
        
    if book_res and candle_res:
        df, price, bids, asks = process_data(book_res, candle_res)
        matches, current = find_patterns(df)
        buy_walls, sell_walls = get_walls(bids, asks, price)
        var_95, upside, vol = run_risk(price, df, granularity/60)
        
        if matches:
            avg_move = np.mean([m['pct'] for m in matches])
            win_rate = sum(1 for m in matches if m['pct'] > 0) / len(matches) * 100
        else: avg_move, win_rate = 0, 0

        # --- PREMIUM METRIC CARDS ---
        col1, col2, col3, col4 = st.columns(4)
        
        # Helper to make cards
        def card(col, label, value, sub="", color="white"):
            col.markdown(f"""
            <div style="background:#161b22; padding:15px; border-radius:10px; border:1px solid #30363d; text-align:center;">
                <p style="color:#8b949e; font-size:12px; margin:0;">{label}</p>
                <p style="color:{color}; font-size:24px; font-weight:bold; margin:5px 0;">{value}</p>
                <p style="color:#8b949e; font-size:10px; margin:0;">{sub}</p>
            </div>
            """, unsafe_allow_html=True)

        risk_color = "#ff5252" if vol > 0.005 else "#69f0ae"
        fractal_color = "#69f0ae" if avg_move > 0 else "#ff5252"

        card(col1, "CURRENT PRICE", f"${price:,.2f}", "Live Data")
        card(col2, "VOLATILITY", f"{vol*100:.2f}%", f"Risk Level: {'High' if vol>0.005 else 'Low'}", risk_color)
        card(col3, "HISTORICAL WIN RATE", f"{win_rate:.0f}%", f"{len(matches)} Matches Found")
        card(col4, "FRACTAL FORECAST", f"{avg_move:+.2f}%", "Next 10 Periods", fractal_color)

        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["🔮 Pattern Match", "📅 Seasonality (IST)", "🎲 Risk"])
        
        with tab1:
            if matches:
                match_date = matches[0]['date'].tz_convert('Asia/Kolkata').strftime('%Y-%m-%d')
                st.caption(f"Top Match Date: {match_date} IST")
                st.plotly_chart(plot_fractal_chart(current, matches[0], buy_walls, sell_walls), use_container_width=True)
            else: st.warning("No clear history matches.")
                
        with tab2:
            if df_season is not None:
                h_stats, d_stats = calculate_simple_seasonality(df_season)
                c_day, c_hour = st.columns(2)
                with c_day: st.plotly_chart(plot_bar(d_stats, "Best Day"), use_container_width=True)
                with c_hour: st.plotly_chart(plot_bar(h_stats, "Best Hour (IST)"), use_container_width=True)
            else: st.error("Could not fetch seasonality data.")
                
        with tab3:
            st.subheader(f"Statistical Range ({tf_label})")
            r1, r2 = st.columns(2)
            card(r1, "STOP LOSS (VaR 95%)", f"${var_95:,.2f}", "Downside Protection", "#ff5252")
            card(r2, "TAKE PROFIT (Upside 95%)", f"${upside:,.2f}", "Upside Target", "#69f0ae")

    else: st.error("API Error.")
