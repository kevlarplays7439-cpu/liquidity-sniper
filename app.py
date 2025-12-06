import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
import time

# --- CONFIG ---
st.set_page_config(page_title="FractalSearch Engine", page_icon="⏳", layout="wide")
st.markdown("""
    <style>
    .metric-card { background-color: #0E1117; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA ENGINE ---
@st.cache_data(ttl=3600) # Cache for 1 hour to speed up searching
def get_historical_data(symbol, period="2y", interval="1h"):
    """Fetches long-term history to search through."""
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return None
        # Clean data
        df = df[['Close']].reset_index()
        df.columns = ['time', 'close']
        return df
    except: return None

# --- 2. MATH ENGINE (PATTERN MATCHING) ---
def normalize(series):
    """Normalizes price data to a 0-1 scale so $20k BTC can match $90k BTC."""
    min_val = np.min(series)
    max_val = np.max(series)
    if max_val - min_val == 0: return series
    return (series - min_val) / (max_val - min_val)

def find_similar_patterns(df, lookback=50, top_k=3):
    """
    The Core Algo: Scans history for patterns similar to the current one.
    """
    # 1. Get the Current Pattern (The Target)
    current_pattern = df['close'].tail(lookback).values
    norm_target = normalize(current_pattern)
    
    matches = []
    prices = df['close'].values
    dates = df['time'].values
    
    # 2. Sliding Window Search (Scan history)
    # We stop 'lookback' steps before the end to avoid matching with itself
    history_len = len(prices) - lookback * 2 
    
    progress_bar = st.progress(0)
    
    # Scan every window
    for i in range(0, history_len):
        # Update progress bar every 10%
        if i % (history_len // 10) == 0: progress_bar.progress(i / history_len)
            
        # Extract candidate pattern
        candidate = prices[i : i + lookback]
        norm_candidate = normalize(candidate)
        
        # Calculate Similarity (Pearson Correlation)
        # 1.0 = Identical, 0.0 = Random, -1.0 = Inverse
        correlation, _ = pearsonr(norm_target, norm_candidate)
        
        # If highly similar, store it
        if correlation > 0.80:
            # Check what happened NEXT (The Future)
            # We look 20 candles into the future from that point
            future_start = i + lookback
            future_end = i + lookback + 20
            
            if future_end < len(prices):
                future_outcome = prices[future_start : future_end]
                
                # Calculate profit/loss of that future
                entry = prices[future_start-1]
                exit = prices[future_end-1]
                pct_change = ((exit - entry) / entry) * 100
                
                matches.append({
                    "date": dates[i],
                    "correlation": correlation,
                    "pattern": candidate,
                    "future": future_outcome,
                    "outcome_pct": pct_change,
                    "start_idx": i
                })
    
    progress_bar.empty()
    
    # 3. Sort by highest similarity
    matches = sorted(matches, key=lambda x: x['correlation'], reverse=True)
    return matches[:top_k], current_pattern

# --- 3. CHART ENGINE ---
def plot_fractal(current_pattern, match_data):
    fig = go.Figure()
    
    # A. Current Market (The "Now")
    x_current = list(range(len(current_pattern)))
    norm_curr = normalize(current_pattern)
    
    fig.add_trace(go.Scatter(
        x=x_current, y=norm_curr,
        mode='lines', name='Current Market',
        line=dict(color='#00FF00', width=3)
    ))
    
    # B. Historical Match (The "Past")
    match_pattern = match_data['pattern']
    future_pattern = match_data['future']
    full_sequence = np.concatenate([match_pattern, future_pattern])
    
    # Normalize historical match to fit the same visual scale
    norm_hist = normalize(full_sequence)
    
    # Split into "Pattern" (Solid) and "Future" (Dotted)
    x_hist = list(range(len(norm_hist)))
    
    # The Pattern that matched
    fig.add_trace(go.Scatter(
        x=x_hist[:len(match_pattern)], 
        y=norm_hist[:len(match_pattern)],
        mode='lines', name=f"History ({match_data['date'].strftime('%Y-%m-%d')})",
        line=dict(color='gray', width=2, dash='dot')
    ))
    
    # The Future (What happened next)
    outcome_color = "#00c853" if match_data['outcome_pct'] > 0 else "#d50000"
    fig.add_trace(go.Scatter(
        x=x_hist[len(match_pattern)-1:], 
        y=norm_hist[len(match_pattern)-1:],
        mode='lines', name=f"Outcome ({match_data['outcome_pct']:.2f}%)",
        line=dict(color=outcome_color, width=3)
    ))
    
    fig.add_vline(x=len(current_pattern)-1, line_dash="dash", annotation_text="TODAY")
    
    fig.update_layout(
        template="plotly_dark", 
        title=f"Fractal Match: {match_data['correlation']*100:.1f}% Similarity",
        height=400, margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Time (Candles)", yaxis_title="Normalized Price (Shape)",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig

# --- 4. APP UI ---
st.sidebar.header("⏳ FractalSearch")
symbol = st.sidebar.text_input("Symbol", "BTC-USD").upper()
lookback = st.sidebar.slider("Pattern Length (Candles)", 30, 100, 50)
st.sidebar.caption("Scans last 2 Years of Hourly Data")

st.title(f"🔍 {symbol} Pattern Recognition Engine")
st.caption("Finds historical moments that look exactly like today to predict tomorrow.")

# Main Execution
if st.button("🚀 Scan History"):
    with st.spinner(f"Downloading 2 years of {symbol} data..."):
        df = get_historical_data(symbol)
    
    if df is not None:
        with st.spinner(f"Scanning {len(df)} historical candles for matches..."):
            matches, current_pattern = find_similar_patterns(df, lookback=lookback)
        
        if not matches:
            st.warning("No high-confidence matches found. Try reducing pattern length.")
        else:
            # SUMMARY STATISTICS
            bullish_count = sum(1 for m in matches if m['outcome_pct'] > 0)
            avg_outcome = np.mean([m['outcome_pct'] for m in matches])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Top Matches Found", len(matches))
            col2.metric("Bullish Probability", f"{bullish_count/len(matches)*100:.0f}%")
            col3.metric("Avg Predicted Move", f"{avg_outcome:+.2f}%", 
                        delta_color="normal" if avg_outcome > 0 else "inverse")
            
            st.divider()
            
            # SHOW MATCHES
            for i, match in enumerate(matches):
                st.subheader(f"#{i+1} Match: {match['date'].strftime('%d %b %Y')}")
                st.plotly_chart(plot_fractal(current_pattern, match), use_container_width=True)
                
                with st.expander("See Data Details"):
                    st.write(f"**Correlation:** {match['correlation']:.4f}")
                    st.write(f"**What happened next:** Price moved {match['outcome_pct']:.2f}% in the following 20 hours.")
    else:
        st.error("Could not fetch data. Check symbol.")
