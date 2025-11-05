import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Real Economic Dashboard",
    page_icon="🌍",
    layout="wide"
)

class RealEconomicData:
    def __init__(self):
        # Safely get API keys from Streamlit secrets
        try:
            self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
            self.fred_key = st.secrets.get("FRED_KEY", "demo")
        except:
            # Fallback for local development
            self.alpha_vantage_key = "demo"
            self.fred_key = "demo"
        
        # Show status
        if self.alpha_vantage_key != "demo":
            st.sidebar.success("✅ Using real API keys")
        else:
            st.sidebar.warning("⚠️ Using demo keys - get real keys for better data")
        
    def get_stock_data(self, symbol):
        """Get real stock data from Alpha Vantage"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                time_series = data.get('Time Series (Daily)', {})
                
                dates = []
                prices = []
                
                for date, values in time_series.items():
                    dates.append(datetime.strptime(date, '%Y-%m-%d'))
                    prices.append(float(values['4. close']))
                
                df = pd.DataFrame({'date': dates, 'price': prices})
                df = df.sort_values('date').reset_index(drop=True)
                return df
                
        except Exception as e:
            st.warning(f"Could not fetch stock data: {e}")
            
        return None
    
    def get_crypto_data(self, symbol="BTC"):
        """Get real cryptocurrency data"""
        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'DIGITAL_CURRENCY_DAILY',
                'symbol': symbol,
                'market': 'USD',
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                time_series = data.get('Time Series (Digital Currency Daily)', {})
                
                dates = []
                prices = []
                
                for date, values in time_series.items():
                    dates.append(datetime.strptime(date, '%Y-%m-%d'))
                    prices.append(float(values['4a. close (USD)']))
                
                df = pd.DataFrame({'date': dates, 'price': prices})
                df = df.sort_values('date').reset_index(drop=True)
                return df
                
        except Exception as e:
            st.warning(f"Could not fetch crypto data: {e}")
            
        return None
    
    def get_economic_indicators(self):
        """Get real economic indicators from various free APIs"""
        try:
            # World Bank API for GDP data (free, no key needed)
            url = "http://api.worldbank.org/v2/country/USA/indicator/NY.GDP.MKTP.CD?format=json&per_page=10"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    gdp_data = []
                    for item in data[1]:
                        if item['value']:
                            gdp_data.append({
                                'year': int(item['date']),
                                'gdp': float(item['value']) / 1e9  # Convert to billions
                            })
                    
                    return pd.DataFrame(gdp_data)
                    
        except Exception as e:
            st.warning(f"Could not fetch economic indicators: {e}")
            
        return None
    
    def get_inflation_data(self):
        """Get inflation data from free API"""
        try:
            # Using World Bank inflation data
            url = "http://api.worldbank.org/v2/country/USA/indicator/FP.CPI.TOTL?format=json&per_page=20"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    inflation_data = []
                    for item in data[1]:
                        if item['value']:
                            inflation_data.append({
                                'year': int(item['date']),
                                'inflation': float(item['value'])
                            })
                    
                    df = pd.DataFrame(inflation_data)
                    # Calculate inflation rate
                    df['inflation_rate'] = df['inflation'].pct_change() * 100
                    return df
                    
        except Exception as e:
            st.warning(f"Could not fetch inflation data: {e}")
            
        return None

def generate_forecast(df, target_column, periods=30):
    """Generate forecasts using multiple models"""
    if df is None or len(df) < 10:
        return None
        
    try:
        # Prepare data
        df = df.sort_values('date').reset_index(drop=True)
        df['days'] = (df['date'] - df['date'].min()).dt.days
        
        X = df[['days']].values
        y = df[target_column].values
        
        # Linear Regression forecast
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        
        # Random Forest forecast
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Generate future dates
        last_date = df['date'].max()
        future_dates = [last_date + timedelta(days=i) for i in range(1, periods + 1)]
        future_days = [(date - df['date'].min()).days for date in future_dates]
        
        # Predictions
        lr_predictions = lr_model.predict(np.array(future_days).reshape(-1, 1))
        rf_predictions = rf_model.predict(np.array(future_days).reshape(-1, 1))
        
        # Ensemble (average of both models)
        ensemble_predictions = (lr_predictions + rf_predictions) / 2
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            'linear_forecast': lr_predictions,
            'random_forest_forecast': rf_predictions,
            'ensemble_forecast': ensemble_predictions
        })
        
        return forecast_df
        
    except Exception as e:
        st.error(f"Forecast generation failed: {e}")
        return None

# Initialize data fetcher
data_fetcher = RealEconomicData()

st.title("🌍 Real-Time Economic Dashboard")
st.markdown("### Live Data & AI Forecasting")

# Sidebar for controls
st.sidebar.header("📊 Data Sources")
data_source = st.sidebar.selectbox(
    "Select Data Source",
    ["Stock Market", "Cryptocurrency", "Economic Indicators", "Inflation Data"]
)

# Main dashboard
if data_source == "Stock Market":
    st.header("📈 Stock Market Analysis")
    
    stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
    
    if st.button("Get Stock Data"):
        with st.spinner("Fetching real stock data..."):
            stock_data = data_fetcher.get_stock_data(stock_symbol)
            
            if stock_data is not None:
                # Display current price
                current_price = stock_data['price'].iloc[-1]
                previous_price = stock_data['price'].iloc[-2]
                change = ((current_price - previous_price) / previous_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{stock_symbol} Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Daily Change", f"{change:.2f}%")
                with col3:
                    st.metric("Data Points", len(stock_data))
                
                # Plot historical data
                fig = px.line(stock_data, x='date', y='price', 
                              title=f'{stock_symbol} Price History')
                st.plotly_chart(fig, use_container_width=True)
                
                # Generate and display forecast
                st.subheader("🔮 Price Forecast (Next 30 Days)")
                forecast = generate_forecast(stock_data, 'price', 30)
                
                if forecast is not None:
                    # Combine historical and forecast data
                    historical_df = stock_data[['date', 'price']].copy()
                    historical_df['type'] = 'historical'
                    
                    forecast_df = forecast[['date', 'ensemble_forecast']].copy()
                    forecast_df = forecast_df.rename(columns={'ensemble_forecast': 'price'})
                    forecast_df['type'] = 'forecast'
                    
                    combined_df = pd.concat([historical_df, forecast_df])
                    
                    # Plot combined data
                    fig2 = px.line(combined_df, x='date', y='price', color='type',
                                  title=f'{stock_symbol} Price Forecast')
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    # Forecast metrics
                    last_historical = historical_df['price'].iloc[-1]
                    final_forecast = forecast_df['price'].iloc[-1]
                    total_change = ((final_forecast - last_historical) / last_historical) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${last_historical:.2f}")
                    with col2:
                        st.metric("30-Day Forecast", f"${final_forecast:.2f}")
                    with col3:
                        st.metric("Expected Change", f"{total_change:.2f}%")
            else:
                st.error("Could not fetch stock data. Please check the symbol or try again later.")

elif data_source == "Cryptocurrency":
    st.header("₿ Cryptocurrency Analysis")
    
    crypto_symbol = st.selectbox("Select Cryptocurrency", 
                                ["BTC", "ETH", "ADA", "DOT", "DOGE"])
    
    if st.button("Get Crypto Data"):
        with st.spinner("Fetching cryptocurrency data..."):
            crypto_data = data_fetcher.get_crypto_data(crypto_symbol)
            
            if crypto_data is not None:
                current_price = crypto_data['price'].iloc[-1]
                previous_price = crypto_data['price'].iloc[-2]
                change = ((current_price - previous_price) / previous_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{crypto_symbol} Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Daily Change", f"{change:.2f}%")
                with col3:
                    st.metric("Volatility", 
                             f"{crypto_data['price'].pct_change().std() * 100:.2f}%")
                
                # Plot crypto data
                fig = px.line(crypto_data, x='date', y='price',
                              title=f'{crypto_symbol} Price History')
                st.plotly_chart(fig, use_container_width=True)
                
                # Crypto forecast
                st.subheader("🔮 Crypto Price Forecast")
                forecast = generate_forecast(crypto_data, 'price', 30)
                
                if forecast is not None:
                    historical_df = crypto_data[['date', 'price']].copy()
                    historical_df['type'] = 'historical'
                    
                    forecast_df = forecast[['date', 'ensemble_forecast']].copy()
                    forecast_df = forecast_df.rename(columns={'ensemble_forecast': 'price'})
                    forecast_df['type'] = 'forecast'
                    
                    combined_df = pd.concat([historical_df, forecast_df])
                    
                    fig2 = px.line(combined_df, x='date', y='price', color='type',
                                  title=f'{crypto_symbol} Price Forecast')
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("Could not fetch cryptocurrency data. Please try again later.")

elif data_source == "Economic Indicators":
    st.header("📊 Economic Indicators")
    
    if st.button("Load Economic Data"):
        with st.spinner("Fetching economic data from World Bank..."):
            economic_data = data_fetcher.get_economic_indicators()
            
            if economic_data is not None:
                st.subheader("US GDP Growth")
                
                # Calculate GDP growth
                economic_data['gdp_growth'] = economic_data['gdp'].pct_change() * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_gdp = economic_data['gdp'].iloc[-1]
                    st.metric("Current GDP", f"${current_gdp:,.0f}B")
                with col2:
                    growth = economic_data['gdp_growth'].iloc[-1]
                    st.metric("GDP Growth", f"{growth:.2f}%")
                with col3:
                    st.metric("Data Years", len(economic_data))
                
                # GDP chart
                fig = px.line(economic_data, x='year', y='gdp',
                              title='US GDP Over Time')
                st.plotly_chart(fig, use_container_width=True)
                
                # GDP growth chart
                fig2 = px.bar(economic_data, x='year', y='gdp_growth',
                             title='US GDP Growth Rate')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("Could not fetch economic data. Please try again later.")

elif data_source == "Inflation Data":
    st.header("💰 Inflation Analysis")
    
    if st.button("Load Inflation Data"):
        with st.spinner("Fetching inflation data..."):
            inflation_data = data_fetcher.get_inflation_data()
            
            if inflation_data is not None:
                st.subheader("US Inflation Rate")
                
                current_inflation = inflation_data['inflation_rate'].iloc[-1]
                previous_inflation = inflation_data['inflation_rate'].iloc[-2]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Inflation", f"{current_inflation:.2f}%")
                with col2:
                    change = current_inflation - previous_inflation
                    st.metric("Change", f"{change:.2f}%")
                with col3:
                    st.metric("Average Inflation", 
                             f"{inflation_data['inflation_rate'].mean():.2f}%")
                
                # Inflation chart
                fig = px.line(inflation_data, x='year', y='inflation_rate',
                              title='US Inflation Rate Over Time')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Could not fetch inflation data. Please try again later.")

# API Key Information
st.sidebar.markdown("---")
st.sidebar.header("🔑 API Setup")
st.sidebar.info(
    "**For higher rate limits:**\n\n"
    "1. Get free API keys from:\n"
    "   - [Alpha Vantage](https://www.alphavantage.co)\n"
    "   - [FRED](https://fred.stlouisfed.org)\n\n"
    "2. Add to Streamlit Cloud secrets"
)

st.success("🚀 Dashboard Ready! Select a data source above to get started.")
