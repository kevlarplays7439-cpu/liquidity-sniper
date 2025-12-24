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
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gramdev AI Dashboard", layout="wide")

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
    for key in YAHOO_MAP.keys():
        if key.upper() in name.upper(): return key
    return name

# --- 2. LIVE DATA FETCHER ---
def fetch_live_data_forced(yahoo_symbol):
    try:
        df = yf.download(yahoo_symbol, period="3mo", progress=False)
        if df.empty: return None, "No data found."
        
        df = df.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        return df, "Success"
    except Exception as e:
        return None, str(e)

# --- 3. LOAD DATA ---
@st.cache_data
def load_csvs():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
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

scores_df, fund_df, price_df = load_csvs()
if scores_df is None: st.stop()

# --- INIT SESSION STATE ---
if 'live_data_cache' not in st.session_state:
    st.session_state['live_data_cache'] = {}

# ==========================================
# 🚀 SIDEBAR (CONTROLS ARE HERE NOW)
# ==========================================
st.sidebar.title("⚡ Gramdev Controls")

# 1. Navigation
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Forecasting", "⚖️ Portfolio"])

st.sidebar.markdown("---")
st.sidebar.header("🔴 Live Data Manager")

# 2. Global Ticker Selector
valid_tickers = sorted(scores_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Active Stock", valid_tickers)

# 3. Yahoo Symbol Verification
default_yahoo = YAHOO_MAP.get(selected_ticker, f"{selected_ticker}.NS")
yahoo_symbol = st.sidebar.text_input("Yahoo Symbol", default_yahoo)

# 4. THE FETCH BUTTON (ALWAYS VISIBLE)
if st.sidebar.button("Fetch Live Data 🔄"):
    with st.spinner(f"Connecting to {yahoo_symbol}..."):
        live_data, status = fetch_live_data_forced(yahoo_symbol)
        if live_data is not None:
            st.session_state['live_data_cache'][selected_ticker] = live_data
            st.sidebar.success("✅ Updated!")
        else:
            st.sidebar.error(f"❌ Failed: {status}")

# ==========================================
# 📄 MAIN PAGES
# ==========================================

# Prepare Data (Check Memory)
static_data = price_df[price_df['Ticker'] == selected_ticker].sort_values('Date')

if selected_ticker in st.session_state['live_data_cache']:
    # Merge Logic
    live_part = st.session_state['live_data_cache'][selected_ticker]
    live_part['Ticker'] = selected_ticker
    active_df = pd.concat([static_data, live_part], ignore_index=True)
    active_df = active_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
    data_source_text = "🟢 LIVE DATA ACTIVE"
else:
    active_df = static_data
    data_source_text = "⚠️ USING OLD CSV DATA"

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard":
    st.title(f"📊 {selected_ticker} Overview")
    st.caption(data_source_text)
    
    # Metric Logic
    latest_price = active_df['Close'].iloc[-1]
    last_date = active_df['Date'].iloc[-1].strftime('%d-%b-%Y')
    
    sub_f = fund_df[fund_df['Ticker'] == selected_ticker]
    score = scores_df[scores_df['Ticker'] == selected_ticker]['Moat_Score'].values[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Latest Close", f"₹{latest_price:,.2f}", f"Date: {last_date}")
    c1.metric("Moat Score", f"{score}/100")
    
    if not sub_f.empty:
        sales_col = 'Sales' if 'Sales' in sub_f.columns else sub_f.columns[2]
        c2.metric("Latest Sales", f"₹{sub_f.iloc[-1][sales_col]:,.2f} Cr")
    
    fig = px.line(active_df, x='Date', y='Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Forecasting":
    st.title(f"🔮 AI Forecast: {selected_ticker}")
    st.caption(data_source_text)
    
    if st.button("Run LSTM Model"):
        if len(active_df) < 60:
            st.error("Not enough data to forecast.")
        else:
            with st.spinner("Training AI Model..."):
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
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                
                last_60 = scaled[-lookback:].reshape(1, lookback, 1)
                pred = scaler.inverse_transform(model.predict(last_60))[0][0]
                
                last_date = active_df['Date'].iloc[-1]
                next_date = last_date + pd.Timedelta(days=1)
                if next_date.weekday() == 5: next_date += pd.Timedelta(days=2)
                elif next_date.weekday() == 6: next_date += pd.Timedelta(days=1)
                
                st.success(f"🧠 Prediction for {next_date.strftime('%d-%b-%Y')}: ₹{pred:.2f}")

# --- PAGE 3: PORTFOLIO ---
elif page == "⚖️ Portfolio":
    st.title("⚖️ Portfolio Optimization")
    st.info("Note: Portfolio uses stored CSV data for speed.")
    
    multi_select = st.multiselect("Select Stocks", valid_tickers, default=valid_tickers[:3])
    
    if len(multi_select) >= 3:
        if st.button("Optimize Allocation"):
            pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[multi_select].dropna()
            returns = pivot.pct_change().dropna()
            
            mu = returns.mean() * 252
            cov = returns.cov() * 252
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(multi_select)))
            init = [1/len(multi_select)]*len(multi_select)
            
            res = minimize(lambda w: -(np.sum(mu*w)/np.sqrt(np.dot(w.T,np.dot(cov,w)))), init, bounds=bounds, constraints=cons)
            
            df_res = pd.DataFrame({'Stock': multi_select, 'Weight': res.x})
            df_res['Weight'] = df_res['Weight'].apply(lambda x: f"{x*100:.1f}%")
            
            c1, c2 = st.columns(2)
            c1.table(df_res)
            c2.plotly_chart(px.pie(values=res.x, names=multi_select, title="Allocation"))
