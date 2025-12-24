import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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
st.set_page_config(page_title="Gramdev Force Update", layout="wide")

# --- 1. MAPPING DICTIONARY ---
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

def normalize_ticker(name):
    # Helps match CSV 'Ticker' column to our keys
    for key in YAHOO_MAP.keys():
        if key.upper() in name.upper(): return key
    return name

# --- 2. LIVE DATA FETCHER (NO CACHE - FORCED) ---
def fetch_live_data_forced(yahoo_symbol):
    """
    Directly hits Yahoo Finance. No Caching.
    """
    try:
        # Download last 3 months to be safe
        df = yf.download(yahoo_symbol, period="3mo", progress=False)
        
        if df.empty:
            return None, "No data found."
            
        # Clean Yahoo Data
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        
        return df, "Success"
    except Exception as e:
        return None, str(e)

# --- 3. LOAD STATIC CSVs ---
@st.cache_data
def load_csvs():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
        # Apply normalization
        scores['Ticker'] = scores['Ticker'].apply(normalize_ticker)
        fund['Ticker'] = fund['Ticker'].apply(normalize_ticker)
        price['Ticker'] = price['Ticker'].apply(normalize_ticker)
        
        fund['Date'] = pd.to_datetime(fund['Date'])
        price['Date'] = pd.to_datetime(price['Date'])
        
        if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
        if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)
        
        return scores, fund, price
    except:
        return None, None, None

# --- APP START ---
scores_df, fund_df, price_df = load_csvs()
if scores_df is None: st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚡ Gramdev Controller")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🔮 Forecasting", "⚖️ Portfolio"])

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard":
    st.title("📊 Live Data Check")
    
    # 1. Select Internal Name
    valid_tickers = sorted(scores_df['Ticker'].unique())
    ticker = st.selectbox("1. Select Company from CSV", valid_tickers)
    
    # 2. Auto-Guess Yahoo Symbol
    default_yahoo = YAHOO_MAP.get(ticker, f"{ticker}.NS")
    
    # 3. MANUAL OVERRIDE BOX
    yahoo_symbol = st.text_input("2. Verify Yahoo Symbol (Change if needed)", default_yahoo)
    
    # 4. LOAD BUTTON
    if st.button("🔴 FETCH LIVE DATA NOW"):
        with st.spinner(f"Pulling fresh data for {yahoo_symbol}..."):
            # Get Static Data
            static_subset = price_df[price_df['Ticker'] == ticker].sort_values('Date')
            
            # Get Live Data
            live_subset, status = fetch_live_data_forced(yahoo_symbol)
            
            if live_subset is not None:
                # Merge
                live_subset['Ticker'] = ticker
                combined = pd.concat([static_subset, live_subset], ignore_index=True)
                combined = combined.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
                
                # Metrics
                latest_date = combined['Date'].iloc[-1].strftime('%d-%b-%Y')
                latest_price = combined['Close'].iloc[-1]
                
                st.success(f"✅ Data Updated! Latest Date: **{latest_date}**")
                
                c1, c2 = st.columns(2)
                c1.metric("Latest Close Price", f"₹{latest_price:,.2f}")
                c1.metric("Previous Close", f"₹{combined['Close'].iloc[-2]:,.2f}")
                
                # Chart
                fig = px.line(combined, x='Date', y='Close', title=f"{ticker} (Includes Live Data)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"❌ Failed to fetch from Yahoo: {status}")
                st.warning("Showing old CSV data only.")
                fig = px.line(static_subset, x='Date', y='Close', title=f"{ticker} (Old Data Only)")
                st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Forecasting":
    st.title("🔮 AI Forecasting (Live)")
    
    ticker = st.selectbox("Select Stock", sorted(price_df['Ticker'].unique()))
    yahoo_symbol = st.text_input("Yahoo Symbol", YAHOO_MAP.get(ticker, f"{ticker}.NS"))
    
    if st.button("Run AI Forecast"):
        # Fetch Data
        static_subset = price_df[price_df['Ticker'] == ticker].sort_values('Date')
        live_subset, _ = fetch_live_data_forced(yahoo_symbol)
        
        if live_subset is not None:
            combined = pd.concat([static_subset, live_subset], ignore_index=True).drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
            active_df = combined
        else:
            active_df = static_subset
            
        if len(active_df) < 60:
            st.error("Not enough data.")
        else:
            # LSTM Logic
            with st.spinner("Training LSTM..."):
                data = active_df['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(data)
                
                X, y = [], []
                lookback = 60
                for i in range(lookback, len(scaled)):
                    X.append(scaled[i-lookback:i, 0])
                    y.append(scaled[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=5, batch_size=32, verbose=0) # Increased Epochs for better result
                
                last_60 = scaled[-lookback:].reshape(1, lookback, 1)
                pred = scaler.inverse_transform(model.predict(last_60))[0][0]
                
                # Date Logic
                last_date = active_df['Date'].iloc[-1]
                next_date = last_date + pd.Timedelta(days=1)
                if next_date.weekday() == 5: next_date += pd.Timedelta(days=2)
                elif next_date.weekday() == 6: next_date += pd.Timedelta(days=1)
                
                st.success(f"🧠 Forecast for {next_date.strftime('%d-%b-%Y')}: ₹{pred:.2f}")
                st.info(f"Based on data up to: {last_date.strftime('%d-%b-%Y')}")

# --- PAGE 3: PORTFOLIO ---
elif page == "⚖️ Portfolio":
    st.title("⚖️ Portfolio Optimization")
    st.info("Uses local CSV data for speed.")
    
    tickers = sorted(price_df['Ticker'].unique())
    selection = st.multiselect("Select Stocks", tickers, default=tickers[:3])
    
    if len(selection) >= 3:
        if st.button("Optimize"):
            pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selection].dropna()
            returns = pivot.pct_change().dropna()
            
            mu = returns.mean() * 252
            cov = returns.cov() * 252
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(selection)))
            init = [1/len(selection)]*len(selection)
            
            res = minimize(lambda w: -(np.sum(mu*w)/np.sqrt(np.dot(w.T,np.dot(cov,w)))), init, bounds=bounds, constraints=cons)
            
            df_res = pd.DataFrame({'Stock': selection, 'Weight': res.x})
            df_res['Weight'] = df_res['Weight'].apply(lambda x: f"{x*100:.1f}%")
            st.table(df_res)
