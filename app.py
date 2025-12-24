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

# --- 1. MAPPING & HELPERS ---
TICKER_MAP = {
    "Action": "ACE", "Bharat": "BEL", "Blue_Star": "BLUESTARCO", "Caplin": "CAPLIPOINT",
    "C_D_S_L": "CDSL", "Dr_Lal": "LALPATHLAB", "Dynacons": "DYNPRO", "Dynamic": "DYCL",
    "Frontier": "Frontier_Springs", "Ganesh": "GANESHHOU", "HDFC": "HDFCAMC",
    "I_R_C_T_C": "IRCTC", "Indiamart": "INDIAMART", "Indo_Tech": "INDOTECH",
    "J_B_Chem": "JBCHEPHARM", "Jai_Balaji": "JAIBALAJI", "Jyoti": "JYOTIRES",
    "KNR": "KNRCON", "Kingfa": "KINGFA", "Kirl": "KIRLPNU", "Macpower": "MACPOWER",
    "Master": "MASTERTR", "Mazagon": "MAZDOCK", "Monarch": "MONARCH", "Newgen": "NEWGEN",
    "Polycab": "POLYCAB", "Prec": "PRECWIRE", "RRP": "RRP_Defense", "Radhika": "RADHIKAJWE",
    "Schaeffler": "SCHAEFFLER", "Shakti": "SHAKTIPUMP", "Shanthi": "SHANTIGEAR",
    "Sharda": "SHARDAMOTR", "Shilchar": "SHILCHAR", "Sika": "SIKA", "Solar": "SOLARINDS",
    "Stylam": "STYLAMIND", "Swaraj": "SWARAJENG", "Tanfac": "Tanfac_Inds", "Tata": "TATAELXSI",
    "Timex": "TIMEX", "Voltamp": "VOLTAMP", 
    "BLS": "BLS", "Apar": "APARINDS", "Ashoka": "ASHOKA", "Astrazeneca": "ASTRAZEN", 
    "BSE": "BSE", "Cams": "CAMS", "3B": "3B_Blackbio"
}

def normalize_ticker(name):
    if name in TICKER_MAP.values(): return name
    for key, value in TICKER_MAP.items():
        if key.upper() in name.upper(): return value
    return name

def fetch_live_data(existing_df, ticker, yahoo_symbol):
    """
    Fetches missing data from Yahoo Finance and appends it to the CSV data.
    """
    try:
        # Find the last date in our static CSV
        last_date = existing_df['Date'].max()
        
        # Download new data from last_date + 1 day
        start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        new_data = yf.download(yahoo_symbol, start=start_date, progress=False)
        
        if not new_data.empty:
            # Format new data to match our DataFrame
            new_data = new_data.reset_index()
            # Handle potential MultiIndex columns from yfinance
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.get_level_values(0)
            
            # Select and rename columns
            new_data = new_data[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']]
            new_data['Ticker'] = ticker # Add ticker column
            
            # Ensure Date types match
            new_data['Date'] = pd.to_datetime(new_data['Date'])
            
            # Combine
            combined = pd.concat([existing_df, new_data], ignore_index=True)
            return combined, len(new_data)
        
        return existing_df, 0
    except Exception as e:
        return existing_df, -1

# --- 2. LOAD DATA ---
@st.cache_data
def load_base_data():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
        scores['Ticker'] = scores['Ticker'].apply(normalize_ticker)
        fund['Ticker'] = fund['Ticker'].apply(normalize_ticker)
        
        fund['Date'] = pd.to_datetime(fund['Date'])
        price['Date'] = pd.to_datetime(price['Date'])
        
        if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
        if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)
        
        return scores, fund, price
    except FileNotFoundError:
        return None, None, None

# --- MAIN APP LOGIC ---
scores_df, fund_df, price_df = load_base_data()
if scores_df is None: 
    st.error("❌ Critical Error: Data files not found.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🚀 Gramdev Dynamic")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔮 Phase A: AI Forecasting", "⚖️ Phase B: Portfolio Mgmt"])

st.sidebar.markdown("---")
st.sidebar.header("🔴 Live Data Connection")
use_live = st.sidebar.checkbox("Enable Yahoo Finance", value=True)

# Helper to manage data state
if 'live_price_df' not in st.session_state:
    st.session_state['live_price_df'] = price_df.copy()

# --- PAGE 1: DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Dynamic Executive Summary")
    
    # 1. Select Company
    valid_tickers = sorted(scores_df['Ticker'].unique())
    ticker = st.selectbox("Select Company", valid_tickers)
    
    # 2. Yahoo Ticker Mapping
    yahoo_symbol = st.sidebar.text_input("Yahoo Ticker Symbol", f"{ticker}.NS")
    
    # 3. Live Data Update Logic
    active_df = price_df[price_df['Ticker'] == ticker].sort_values('Date')
    
    if use_live:
        with st.spinner(f"Connecting to live market for {ticker}..."):
            updated_df, new_rows = fetch_live_data(active_df, ticker, yahoo_symbol)
            if new_rows > 0:
                st.toast(f"✅ Fetched {new_rows} new days of data!", icon="📈")
            elif new_rows == -1:
                st.error("⚠️ Failed to fetch live data. Check Ticker Symbol.")
            active_df = updated_df

    # 4. Display Metrics
    sub_f = fund_df[fund_df['Ticker'] == ticker].sort_values('Date')
    score = scores_df[scores_df['Ticker'] == ticker]['Moat_Score'].values[0]
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Moat Score", f"{score}/100")
    
    last_price = active_df.iloc[-1]['Close'] if not active_df.empty else 0
    c2.metric("Latest Price", f"₹{last_price:,.2f}")
    
    col = 'Sales' if 'Sales' in sub_f.columns else sub_f.columns[2]
    last_sales = sub_f.iloc[-1][col] if not sub_f.empty else 0
    c3.metric("Latest Sales", f"₹{last_sales:,.2f} Cr")
    
    # 5. Chart
    if not active_df.empty:
        fig = px.line(active_df, x='Date', y='Close', title=f"{ticker} Live Price History")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Phase A: AI Forecasting":
    st.title("🔮 Phase A: Dynamic Forecasting")
    
    ticker = st.selectbox("Select Stock", sorted(price_df['Ticker'].unique()))
    analysis_type = st.radio("Select Analysis Module", ["LSTM Price Forecast", "GARCH Volatility Risk", "ARIMA Trend"])
    
    # Yahoo Ticker Input for Forecast
    yahoo_symbol = st.sidebar.text_input("Yahoo Ticker Symbol", f"{ticker}.NS")
    
    # Data Prep
    active_df = price_df[price_df['Ticker'] == ticker].sort_values('Date')
    if use_live:
        active_df, _ = fetch_live_data(active_df, ticker, yahoo_symbol)
    
    if len(active_df) < 60:
        st.error("Insufficient data.")
    else:
        # Date Logic
        last_date = active_df['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        if next_date.weekday() == 5: next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6: next_date += pd.Timedelta(days=1)
        date_str = next_date.strftime("%d %b %Y")

        if analysis_type == "LSTM Price Forecast":
            st.subheader("🧠 Deep Learning (LSTM)")
            if st.button("Run Neural Network on Live Data"):
                with st.spinner("Training Brain..."):
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
                    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
                    
                    last_60 = scaled[-lookback:].reshape(1, lookback, 1)
                    pred = scaler.inverse_transform(model.predict(last_60))[0][0]
                    
                    st.success(f"🤖 LSTM Prediction for {date_str}: ₹{pred:.2f}")

        elif analysis_type == "GARCH Volatility Risk":
            st.subheader("⚠️ GARCH Volatility Model")
            if st.button("Analyze Live Risk"):
                returns = active_df['Close'].pct_change().dropna() * 100
                am = arch_model(returns, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                st.write(res.summary())
                st.line_chart(res.conditional_volatility)

        elif analysis_type == "ARIMA Trend":
            st.subheader("📈 ARIMA Trend Model")
            model = ARIMA(active_df['Close'], order=(5,1,0))
            fit = model.fit()
            forecast = fit.forecast(steps=1).iloc[0]
            st.info(f"ARIMA Forecast for {date_str}: ₹{forecast:.2f}")

# --- PAGE 3: PORTFOLIO ---
elif page == "⚖️ Phase B: Portfolio Mgmt":
    st.title("⚖️ Phase B: Dynamic Portfolio")
    st.info("Note: Portfolio optimization uses the local CSV data for speed.")
    
    tickers = sorted(price_df['Ticker'].unique())
    selection = st.multiselect("Select Stocks", tickers, default=tickers[:5])
    
    if len(selection) >= 3:
        pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selection].dropna()
        returns = pivot.pct_change().dropna()
        
        tab1, tab2, tab3 = st.tabs(["Clustering", "PCA", "Markowitz Opt"])
        
        with tab1:
            st.subheader("🧬 Stock Clustering")
            k = st.slider("Clusters", 2, 5, 3)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(returns.corr())
            st.table(pd.DataFrame({'Ticker': selection, 'Cluster': kmeans.labels_}).sort_values('Cluster'))
            
        with tab2:
            st.subheader("🧩 PCA Analysis")
            pca = PCA(n_components=3).fit(returns)
            st.bar_chart(pca.explained_variance_ratio_)
            
        with tab3:
            st.subheader("🏆 Optimization")
            if st.button("Optimize Weights"):
                mu = returns.mean() * 252
                cov = returns.cov() * 252
                cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                bounds = tuple((0, 1) for _ in range(len(selection)))
                init = [1/len(selection)]*len(selection)
                res = minimize(lambda w: -(np.sum(mu*w)/np.sqrt(np.dot(w.T,np.dot(cov,w)))), init, bounds=bounds, constraints=cons)
                
                df_res = pd.DataFrame({'Stock': selection, 'Weight': res.x})
                df_res['Weight'] = df_res['Weight'].apply(lambda x: f"{x*100:.1f}%")
                st.table(df_res)
                st.plotly_chart(px.pie(values=res.x, names=selection, title="Optimal Allocation"))
