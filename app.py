import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gramdev Dynamic AI Dashboard", layout="wide")

# --- 1. ROBUST TICKER MAPPING ---
# Maps your CSV names to correct Yahoo Finance Tickers
YAHOO_MAP = {
    "Action": "ACE.NS", "Bharat": "BEL.NS", "Blue_Star": "BLUESTARCO.NS", "Caplin": "CAPLIPOINT.NS",
    "C_D_S_L": "CDSL.NS", "Dr_Lal": "LALPATHLAB.NS", "Dynacons": "DYNPRO.NS", "Dynamic": "DYCL.NS",
    "Frontier": "FRONTIER.BO", "Ganesh": "GANESHHOU.NS", "HDFC": "HDFCAMC.NS",
    "I_R_C_T_C": "IRCTC.NS", "Indiamart": "INDIAMART.NS", "Indo_Tech": "INDOTECH.NS",
    "J_B_Chem": "JBCHEPHARM.NS", "Jai_Balaji": "JAIBALAJI.NS", "Jyoti": "JYOTIRES.NS",
    "KNR": "KNRCON.NS", "Kingfa": "KINGFA.NS", "Kirl": "KIRLPNU.NS", "Macpower": "MACPOWER.NS",
    "Master": "MASTERTR.NS", "Mazagon": "MAZDOCK.NS", "Monarch": "MONARCH.NS", "Newgen": "NEWGEN.NS",
    "Polycab": "POLYCAB.NS", "Prec": "PRECWIRE.NS", "RRP": "RRP.BO", "Radhika": "RADHIKAJWE.BO",
    "Schaeffler": "SCHAEFFLER.NS", "Shakti": "SHAKTIPUMP.NS", "Shanthi": "SHANTIGEAR.NS",
    "Sharda": "SHARDAMOTR.NS", "Shilchar": "SHILCHAR.NS", "Sika": "SIKA.BO", "Solar": "SOLARINDS.NS",
    "Stylam": "STYLAMIND.NS", "Swaraj": "SWARAJENG.NS", "Tanfac": "TANFACIND.NS", "Tata": "TATAELXSI.NS",
    "Timex": "TIMEX.NS", "Voltamp": "VOLTAMP.NS", 
    "BLS": "BLS.NS", "Apar": "APARINDS.NS", "Ashoka": "ASHOKA.NS", "Astrazeneca": "ASTRAZEN.NS", 
    "BSE": "BSE.NS", "Cams": "CAMS.NS", "3B": "3BBLACKBIO.NS"
}

# --- 2. INTERNAL NAME NORMALIZER ---
def normalize_ticker(name):
    # Helps match CSV 'Ticker' column to our keys
    for key in YAHOO_MAP.keys():
        if key.upper() in name.upper(): return key
    # Fallback for exact matches or unknown
    return name

# --- 3. LIVE DATA FETCHER (ROBUST) ---
def fetch_live_data(existing_df, internal_name):
    """
    1. Looks up correct Yahoo Symbol.
    2. Downloads last 30 days (to bridge any gaps).
    3. Merges and De-duplicates.
    """
    # Get correct symbol (or guess .NS)
    symbol = YAHOO_MAP.get(internal_name, f"{internal_name}.NS")
    
    try:
        # Download recent data
        new_data = yf.download(symbol, period="1mo", progress=False)
        
        if new_data.empty:
            return existing_df, 0, f"❌ Failed to find {symbol}"
            
        # Clean Yahoo Data
        new_data = new_data.reset_index()
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.get_level_values(0)
            
        new_data = new_data[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        new_data['Ticker'] = internal_name
        new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None) # Remove timezone for merge
        
        # Merge logic
        existing_df['Date'] = pd.to_datetime(existing_df['Date']).dt.tz_localize(None)
        
        # Combine and Drop Duplicates based on Date
        combined = pd.concat([existing_df, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
        
        # Count how many new rows were actually added
        new_rows = len(combined) - len(existing_df)
        
        status = f"✅ Connected to {symbol}"
        return combined, new_rows, status

    except Exception as e:
        return existing_df, 0, f"⚠️ Error: {str(e)}"

# --- 4. LOAD STATIC DATA ---
@st.cache_data
def load_static_data():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
        # Apply Normalization
        scores['Ticker'] = scores['Ticker'].apply(normalize_ticker)
        fund['Ticker'] = fund['Ticker'].apply(normalize_ticker)
        price['Ticker'] = price['Ticker'].apply(normalize_ticker)
        
        fund['Date'] = pd.to_datetime(fund['Date'])
        price['Date'] = pd.to_datetime(price['Date'])
        
        if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
        if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)
        
        return scores, fund, price
    except FileNotFoundError:
        return None, None, None

# --- MAIN APP ---
scores_df, fund_df, price_df = load_static_data()
if scores_df is None: 
    st.error("❌ Critical Error: CSV files missing.")
    st.stop()

# --- SIDEBAR & NAVIGATION ---
st.sidebar.title("🚀 Gramdev Dynamic")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔮 Phase A: AI Forecasting", "⚖️ Phase B: Portfolio Mgmt"])

st.sidebar.markdown("---")
st.sidebar.header("🔴 Live Connection")
use_live = st.sidebar.checkbox("Sync with Yahoo Finance", value=True)

if st.sidebar.button("🔄 Force Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# --- PAGE 1: DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Summary")
    
    valid_tickers = sorted(scores_df['Ticker'].unique())
    ticker = st.selectbox("Select Company", valid_tickers)
    
    # Get Base Data
    active_df = price_df[price_df['Ticker'] == ticker].sort_values('Date')
    
    # LIVE UPDATE
    status_msg = ""
    if use_live:
        with st.spinner("Syncing..."):
            active_df, new_count, status_msg = fetch_live_data(active_df, ticker)
            
    # Metrics
    sub_f = fund_df[fund_df['Ticker'] == ticker].sort_values('Date')
    score = scores_df[scores_df['Ticker'] == ticker]['Moat_Score'].values[0]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Moat Score", f"{score}/100")
    
    last_price = active_df.iloc[-1]['Close'] if not active_df.empty else 0
    c2.metric("Latest Price", f"₹{last_price:,.2f}")
    
    col = 'Sales' if 'Sales' in sub_f.columns else sub_f.columns[2]
    c3.metric("Latest Sales", f"₹{sub_f.iloc[-1][col]:,.2f} Cr" if not sub_f.empty else "N/A")
    
    c4.caption(f"Data Source:\n{status_msg}")

    if not active_df.empty:
        fig = px.line(active_df, x='Date', y='Close', title=f"{ticker} Live Price Trend")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Phase A: AI Forecasting":
    st.title("🔮 Phase A: Dynamic Forecasting")
    
    ticker = st.selectbox("Select Stock", sorted(price_df['Ticker'].unique()))
    analysis_type = st.radio("Select Analysis Module", ["LSTM Price Forecast", "GARCH Volatility Risk", "ARIMA Trend"])
    
    # Prepare Data
    active
