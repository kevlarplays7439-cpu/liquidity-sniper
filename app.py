import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import pearsonr
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
    
    /* REMOVE PADDING */
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

    /* GLASSMOPHISM CARDS */
    .metric-card {
        background: rgba(20, 20, 20, 0.7);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00ff7f;
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
    }
    
    /* TYPOGRAPHY */
    .metric-label { color: #888; font-size: 12px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #fff; font-size: 28px; font-weight: 700; margin-top: 5px; }
    .metric-sub { font-size: 12px; margin-top: 5px; font-weight: 500; }
    
    /* CUSTOM TABS */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; background-color: #000000; padding: 5px; border-radius: 30px; border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; border-radius: 20px; color: #888; font-size: 13px; border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f1f1f !important; color: #fff !important;
    }
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

# --- 4. MATH ENGINE ---
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
    df['ema'] = df['close'].ewm(span=50).mean()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss =
