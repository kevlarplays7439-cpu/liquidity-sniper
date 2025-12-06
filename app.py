import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
import yfinance as yf

# --- CONFIG ---
st.set_page_config(page_title="FractalSearch Engine", page_icon="⏳", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINES ---

@st.cache_data(ttl=3600)
def get_data_coinbase(symbol):
    """Good for Crypto (BTC-USD). Works on Cloud."""
    try:
        # Auto-fix symbol for Coinbase (needs dash)
        if "-" not in symbol and "USD" not in symbol: symbol = f"{symbol}-USD"
        
        url = f"https://api.exchange.coinbase.com/products/{symbol}/candles?granularity=86400"
        headers = {"User-Agent": "FractalSearch/1.0"}
        res = requests.get(url, headers=headers, timeout=5).json()
        
        df = pd.DataFrame(res, columns=["time", "low", "high", "open", "close", "vol"])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.sort_values("time").reset_index(drop=True)
        return df
    except: return None

@st.cache_data(ttl=3600)
def get_data_yahoo(symbol):
    """Good for Stocks (NVDA), Forex (EURUSD=X), Gold (GC=F). May fail on Cloud."""
    try:
        # Download 2 years of daily data
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty: return None
        
        # Normalize columns to match Coinbase format
        df = df.reset_index()
        df = df[['Date', 'Close']]
        df.columns = ['time', 'close'] # Rename to standard
        return df
    except: return None

# --- 2. MATH ENGINE ---
def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0: return series
    return (series - min_val) / (max_val - min_val)

def find_similar_patterns(df, lookback=30, top_k=3):
    # Ensure we have enough data
    if len(df) < lookback + 20: return [], []

    current_pattern = df['close'].tail(lookback).values
    norm_target = normalize(current_pattern)
    
    matches = []
    prices = df['close'].values
    dates = df['time'].values
    
    # Search History (Stop before the current pattern starts)
    history_len = len(prices) - lookback - 10 
    
    progress_bar = st.progress(0)
    
    for i in range(0, history_len):
        if i % 50 == 0: progress_bar.progress(i / history_len)
            
        candidate = prices[i : i + lookback]
        norm_candidate = normalize(candidate)
        
        if len(candidate) == len(norm_target):
            # Speed optimization: Only check correlation if endpoints align roughly
            # (Skipping this check for accuracy, using raw Pearson)
            correlation, _ = pearsonr(norm_target, norm_candidate)
            
            if correlation > 0.80:
                future_start = i + lookback
                future_end = i + lookback + 10 
                
                if future_end < len(prices):
                    future = prices[future_start : future_end]
                    entry = prices[future_start-1]
                    exit = prices[future_end-1]
                    pct_change = ((exit - entry) / entry) * 100
                    
                    matches.append({
                        "date": dates[i],
                        "correlation": correlation,
                        "pattern": candidate,
                        "future": future,
                        "outcome_pct": pct_change
                    })
    
    progress_bar.empty()
    matches = sorted(matches, key=lambda x: x['correlation'], reverse=True)
    
    # Filter duplicates (overlapping dates)
    unique_matches = []
    seen_dates = set()
    for m in matches:
        # Simple date check to avoid same week matches
        d_str = str(m['date'])[:7] # Year-Month
        if d_str not in seen_dates:
            unique_matches.append(m)
            seen_dates.add(d_str)
            
    return unique_matches[:top_k], current_pattern

# --- 3. CHART ENGINE ---
def plot_fractal(current_pattern, match_data):
    fig = go.Figure()
    
    # Normalize inputs for visual comparison
    combined = np.concatenate([current_pattern, match_data['pattern'], match_data['future']])
    norm_factor = (np.max(combined) - np.min(combined))
    base = np.min(combined)
    def norm(arr): return (arr - base) / norm_factor

    # Current Market (Green)
    fig.add_trace(go.Scatter(x=list(range(len(current_pattern))), y=norm(current_pattern),
                             mode='lines', name='Current Market', line=dict(color='#00FF00', width=4)))
    
    # Historical Match (Grey)
    fig.add_trace(go.Scatter(x=list(range(len(match_data['pattern']))), y=norm(match_data['pattern']),
                             mode='lines', name=f"History ({match_data['date'].strftime('%Y')})", line=dict(color='gray', width=2, dash='dot')))
    
    # Future Outcome (Red/Green)
    outcome_color = "#00c853" if match_data['outcome_pct'] > 0 else "#d50000"
    start_x = len(current_pattern) - 1
    future_x = list(range(start_x, start_x + len(match_data['future'])))
    
    fig.add_trace(go.Scatter(x=future_x, y=norm(match_data['future']),
                             mode='lines', name='Projected Move', line=dict(color=outcome_color, width=4)))
    
    fig.add_vline(x=len(current_pattern)-1, line_dash="dash", annotation_text="TODAY")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title="Days", yaxis_title="Normalized Price", hovermode="x")
    return fig

# --- 4. MAIN APP ---
st.sidebar.header("⏳ FractalSearch")

# DATA SOURCE SELECTOR
data_source = st.sidebar.radio("Data Source", ["Coinbase (Crypto Only)", "Yahoo Finance (Stocks/Forex)"])

if "Coinbase" in data_source:
    sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()
    st.sidebar.caption("✅ Best for Cloud Reliability")
else:
    sym_input = st.sidebar.text_input("Symbol", "NVDA").upper()
    st.sidebar.caption("⚠️ May fail on Cloud (Works Locally)")

lookback = st.sidebar.slider("Pattern Length (Days)", 14, 60, 30)

st.title(f"🔍 {sym_input} Fractal Engine")
st.caption(f"Scanning history using {data_source}...")

if st.button("🚀 Scan History"):
    df = None
    with st.spinner("Downloading data..."):
        if "Coinbase" in data_source:
            df = get_data_coinbase(sym_input)
        else:
            df = get_data_yahoo(sym_input)
    
    if df is not None:
        with st.spinner(f"Analyzing {len(df)} candles..."):
            matches, current = find_similar_patterns(df, lookback=lookback)
        
        if not matches:
            st.warning("No clear fractal matches found. Try a different timeframe.")
        else:
            # Stats
            avg_return = np.mean([m['outcome_pct'] for m in matches])
            win_rate = sum(1 for m in matches if m['outcome_pct'] > 0) / len(matches) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Matches Found", len(matches))
            c2.metric("Bullish Probability", f"{win_rate:.0f}%")
            c3.metric("Projected Move", f"{avg_return:+.2f}%", delta_color="normal" if avg_return > 0 else "inverse")
            
            st.divider()
            
            for i, m in enumerate(matches):
                st.subheader(f"Match #{i+1}: {m['date'].strftime('%d %b %Y')}")
                st.plotly_chart(plot_fractal(current, m), use_container_width=True)
                with st.expander("Details"):
                    st.write(f"Similarity Score: {m['correlation']*100:.1f}%")
                    st.write(f"Result: {m['outcome_pct']:.2f}% move over next 10 days.")
    else:
        st.error(f"Could not fetch data. If using Yahoo on Cloud, try switching to Coinbase.")
