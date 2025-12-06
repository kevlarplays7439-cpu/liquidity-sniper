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
    try:
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
    try:
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df.empty: return None
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
        if 'Date' in df.columns: df = df.rename(columns={'Date': 'time', 'Close': 'close'})
        df = df[['time', 'close']]
        df['time'] = pd.to_datetime(df['time'])
        return df
    except: return None

# --- 2. MONTE CARLO ENGINE (NEW) ---
def run_monte_carlo(current_price, df_history, simulations=2000, days_forward=10):
    # Calculate daily volatility (Standard Deviation of returns)
    returns = df_history['close'].pct_change().dropna()
    daily_vol = returns.std()
    
    # Scale volatility for the forecast period (Square Root of Time Rule)
    # If looking 10 days ahead, volatility is higher than 1 day ahead
    period_vol = daily_vol * np.sqrt(days_forward)
    
    # Run Simulations
    sim_returns = np.random.normal(0, period_vol, simulations)
    future_prices = current_price * (1 + sim_returns)
    future_prices = np.sort(future_prices)
    
    # 95% Confidence Interval (Value at Risk)
    var_95_price = np.percentile(future_prices, 5)  # Worst case
    upside_95_price = np.percentile(future_prices, 95) # Best case
    
    return var_95_price, upside_95_price, daily_vol

# --- 3. PATTERN ENGINE ---
def normalize(series):
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0: return series
    return (series - min_val) / (max_val - min_val)

def find_similar_patterns(df, lookback=30, top_k=3):
    if len(df) < lookback + 20: return [], []
    current_pattern = df['close'].tail(lookback).values
    norm_target = normalize(current_pattern)
    matches = []
    prices = df['close'].values
    dates = df['time'].values
    history_len = len(prices) - lookback - 10 
    progress_bar = st.progress(0)
    
    for i in range(0, history_len):
        if i % 50 == 0: progress_bar.progress(i / history_len)
        candidate = prices[i : i + lookback]
        if len(candidate) == len(norm_target):
            norm_candidate = normalize(candidate)
            correlation, _ = pearsonr(norm_target, norm_candidate)
            if correlation > 0.80:
                future = prices[i + lookback : i + lookback + 10]
                pct_change = ((future[-1] - prices[i + lookback-1]) / prices[i + lookback-1]) * 100
                matches.append({
                    "date": dates[i], "correlation": correlation,
                    "pattern": candidate, "future": future, "outcome_pct": pct_change
                })
    
    progress_bar.empty()
    matches = sorted(matches, key=lambda x: x['correlation'], reverse=True)
    unique_matches = []
    seen_dates = set()
    for m in matches:
        d_str = str(m['date'])[:7] 
        if d_str not in seen_dates:
            unique_matches.append(m)
            seen_dates.add(d_str)
    return unique_matches[:top_k], current_pattern

# --- 4. CHART ENGINE ---
def plot_fractal(current_pattern, match_data):
    fig = go.Figure()
    combined = np.concatenate([current_pattern, match_data['pattern'], match_data['future']])
    norm_factor = (np.max(combined) - np.min(combined))
    base = np.min(combined)
    def norm(arr): return (arr - base) / norm_factor
    safe_date = pd.to_datetime(match_data['date']).strftime('%Y')

    fig.add_trace(go.Scatter(x=list(range(len(current_pattern))), y=norm(current_pattern),
                             mode='lines', name='Current Market', line=dict(color='#00FF00', width=4)))
    fig.add_trace(go.Scatter(x=list(range(len(match_data['pattern']))), y=norm(match_data['pattern']),
                             mode='lines', name=f"History ({safe_date})", line=dict(color='gray', width=2, dash='dot')))
    
    outcome_color = "#00c853" if match_data['outcome_pct'] > 0 else "#d50000"
    start_x = len(current_pattern) - 1
    future_x = list(range(start_x, start_x + len(match_data['future'])))
    
    fig.add_trace(go.Scatter(x=future_x, y=norm(match_data['future']),
                             mode='lines', name='Projected Move', line=dict(color=outcome_color, width=4)))
    
    fig.add_vline(x=len(current_pattern)-1, line_dash="dash", annotation_text="TODAY")
    fig.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=0),
                      xaxis_title="Days", yaxis_title="Normalized Price", hovermode="x")
    return fig

# --- 5. MAIN APP ---
st.sidebar.header("⏳ FractalSearch")
data_source = st.sidebar.radio("Data Source", ["Coinbase (Crypto)", "Yahoo (Stocks)"])
if "Coinbase" in data_source: sym_input = st.sidebar.text_input("Symbol", "BTC-USD").upper()
else: sym_input = st.sidebar.text_input("Symbol", "NVDA").upper()
lookback = st.sidebar.slider("Pattern Length", 14, 60, 30)
st.sidebar.caption("Includes Monte Carlo Risk Check")

st.title(f"🔍 {sym_input} Fractal + Monte Carlo")

if st.button("🚀 Run Analysis"):
    df = None
    with st.spinner("Crunching data..."):
        if "Coinbase" in data_source: df = get_data_coinbase(sym_input)
        else: df = get_data_yahoo(sym_input)
    
    if df is not None:
        # 1. RUN MONTE CARLO FIRST
        current_price = df['close'].iloc[-1]
        var_95, upside_95, vol = run_monte_carlo(current_price, df)
        
        # 2. RUN FRACTAL SEARCH
        matches, current = find_similar_patterns(df, lookback=lookback)
        
        # 3. DISPLAY RESULTS
        # Top Stats Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Price", f"${current_price:,.2f}")
        c2.metric("Daily Volatility", f"{vol*100:.2f}%")
        
        # Calculate Fractal Win Rate
        if matches:
            win_rate = sum(1 for m in matches if m['outcome_pct'] > 0) / len(matches) * 100
            c3.metric("Historical Win Rate", f"{win_rate:.0f}%")
        else:
            c3.metric("Matches Found", "0")

        st.divider()
        
        # --- RISK ANALYSIS SECTION (NEW) ---
        rc1, rc2 = st.columns(2)
        with rc1:
            st.subheader("🎲 Monte Carlo Forecast (10 Days)")
            st.write(f"Based on **{vol*100:.2f}%** volatility, here is the statistical range for the next 10 days:")
            st.error(f"🛑 **Worst Case (VaR 95%):** ${var_95:,.2f}")
            st.success(f"🚀 **Best Case (Upside 95%):** ${upside_95:,.2f}")
        
        with rc2:
            st.subheader("🔮 Historical Precedent")
            if matches:
                avg_move = np.mean([m['outcome_pct'] for m in matches])
                sentiment = "BULLISH" if avg_move > 0 else "BEARISH"
                color = "green" if avg_move > 0 else "red"
                st.markdown(f"History says: <span style='color:{color}; font-size:24px'>**{sentiment}**</span>", unsafe_allow_html=True)
                st.write(f"Average move after this pattern: **{avg_move:+.2f}%**")
            else:
                st.info("No patterns found to form an opinion.")

        st.divider()
        
        # --- FRACTAL MATCHES ---
        if matches:
            for i, m in enumerate(matches):
                safe_date = pd.to_datetime(m['date']).strftime('%d %b %Y')
                st.subheader(f"Match #{i+1}: {safe_date}")
                st.plotly_chart(plot_fractal(current, m), use_container_width=True)
                with st.expander(f"Details for Match #{i+1}"):
                    st.write(f"Correlation: {m['correlation']:.2f}")
                    st.write(f"Result: {m['outcome_pct']:.2f}%")
    else:
        st.error("Data Error. Try switching source.")
