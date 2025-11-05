import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
from datetime import datetime, timedelta
import time
import json
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, but provide fallback if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Using linear regression for forecasting.")

# Page configuration
st.set_page_config(
    page_title="Real-Time Economic Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealTimeEconomicDashboard:
    def __init__(self):
        # Safe secret handling with fallbacks
        try:
            self.fred_api_key = st.secrets.get("FRED_API_KEY", "DEMO_KEY")
            self.alpha_vantage_key = st.secrets.get("ALPHA_VANTAGE_KEY", "demo")
        except Exception:
            # Fallback for environments without Streamlit secrets
            self.fred_api_key = "DEMO_KEY"
            self.alpha_vantage_key = "demo"
            st.info("Using demo API keys. For real data, add your API keys to Streamlit secrets.")
        
        self.world_bank_url = "http://api.worldbank.org/v2"
        self.prophet_available = PROPHET_AVAILABLE
        
    def get_sample_economic_data(self):
        """Generate realistic sample economic data"""
        np.random.seed(42)
        
        countries = ['USA', 'Canada', 'UK', 'Germany', 'France', 'Japan', 'Australia', 'Brazil', 'India', 'China']
        coordinates = {
            'USA': [-95.7129, 37.0902], 'Canada': [-106.3468, 56.1304],
            'UK': [-3.4360, 55.3781], 'Germany': [10.4515, 51.1657],
            'France': [2.2137, 46.2276], 'Japan': [138.2529, 36.2048],
            'Australia': [133.7751, -25.2744], 'Brazil': [-53.2, -10.3333],
            'India': [78.9629, 20.5937], 'China': [104.1954, 35.8617]
        }
        
        data = []
        current_date = datetime.now()
        
        for country in countries:
            base_gdp = np.random.uniform(500, 5000)
            base_inflation = np.random.normal(0.025, 0.005)
            base_unemployment = np.random.uniform(0.03, 0.08)
            
            for months_back in range(60, -1, -1):  # 5 years of monthly data
                date = current_date - timedelta(days=30*months_back)
                
                # Add some realistic trends and seasonality
                trend = months_back / 600  # Small upward trend
                seasonal = np.sin(2 * np.pi * date.month / 12) * 0.01
                
                gdp = base_gdp * (1 + 0.02/12) ** months_back + np.random.normal(0, 10)
                inflation = base_inflation + seasonal + np.random.normal(0, 0.002)
                unemployment = max(0.01, base_unemployment - trend + np.random.normal(0, 0.005))
                
                data.append({
                    'country': country,
                    'date': date,
                    'gdp': gdp,
                    'inflation': inflation,
                    'unemployment': unemployment,
                    'lat': coordinates[country][1],
                    'lon': coordinates[country][0]
                })
        
        return pd.DataFrame(data)
    
    @st.cache_data(ttl=3600)
    def fetch_fred_data(_self, series_id, country_name):
        """Fetch real-time economic data from FRED API with demo fallback"""
        # If using demo key, return sample data
        if _self.fred_api_key == "DEMO_KEY":
            return _self.generate_sample_fred_data(series_id, country_name)
            
        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': series_id,
                'api_key': _self.fred_api_key,
                'file_type': 'json',
                'observation_start': '2015-01-01'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                observations = data.get('observations', [])
                
                df = pd.DataFrame(observations)
                if not df.empty:
                    df['value'] = pd.to_numeric(df['value'], errors='coerce')
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.dropna(subset=['value'])
                    df['country'] = country_name
                    return df
                    
        except Exception as e:
            st.warning(f"Could not fetch FRED data for {series_id}: {e}")
            
        # Fallback to sample data
        return _self.generate_sample_fred_data(series_id, country_name)
    
    def generate_sample_fred_data(self, series_id, country_name):
        """Generate realistic sample FRED data"""
        dates = pd.date_range(start='2015-01-01', end=datetime.now(), freq='M')
        
        # Different patterns for different series
        if 'GDP' in series_id:
            base_value = 18000
            growth_rate = 0.02
            volatility = 200
        elif 'UNRATE' in series_id:
            base_value = 3.8
            growth_rate = 0
            volatility = 0.3
        elif 'CPI' in series_id:
            base_value = 250
            growth_rate = 0.025
            volatility = 2
        else:
            base_value = 100
            growth_rate = 0.01
            volatility = 5
        
        values = []
        for i, date in enumerate(dates):
            trend = base_value * (1 + growth_rate/12) ** i
            seasonal = np.sin(2 * np.pi * date.month / 12) * volatility * 0.5
            noise = np.random.normal(0, volatility * 0.5)
            value = trend + seasonal + noise
            values.append(value)
        
        return pd.DataFrame({
            'date': dates,
            'value': values,
            'country': country_name
        })
    
    @st.cache_data(ttl=86400)
    def fetch_world_bank_data(_self, indicator, country_codes):
        """Fetch data from World Bank API with fallback"""
        try:
            url = f"{_self.world_bank_url}/country/{country_codes}/indicator/{indicator}"
            params = {'format': 'json', 'per_page': 1000}
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1:
                    records = []
                    for item in data[1]:
                        if item['value'] is not None:
                            records.append({
                                'country': item['country']['value'],
                                'year': int(item['date']),
                                'value': float(item['value']),
                                'indicator': indicator
                            })
                    return pd.DataFrame(records)
                    
        except Exception as e:
            st.warning(f"Could not fetch World Bank data: {e}")
            
        # Fallback to sample data
        return _self.generate_sample_world_bank_data(indicator, country_codes)
    
    def generate_sample_world_bank_data(self, indicator, country_codes):
        """Generate sample World Bank data"""
        countries = country_codes.split(';')
        years = range(2010, 2024)
        
        data = []
        for country in countries:
            country_name = {
                'USA': 'United States', 'CAN': 'Canada', 'GBR': 'United Kingdom',
                'DEU': 'Germany', 'FRA': 'France', 'JPN': 'Japan',
                'AUS': 'Australia', 'BRA': 'Brazil', 'IND': 'India', 'CHN': 'China'
            }.get(country, country)
            
            base_value = np.random.uniform(1000, 5000)
            growth_rate = np.random.normal(0.03, 0.01)
            
            for year in years:
                value = base_value * (1 + growth_rate) ** (year - 2010)
                value += np.random.normal(0, value * 0.05)  # Add some noise
                
                data.append({
                    'country': country_name,
                    'year': year,
                    'value': max(value, 0),  # Ensure positive values
                    'indicator': indicator
                })
        
        return pd.DataFrame(data)
    
    def get_live_market_data(self):
        """Get simulated live market data"""
        # Simulate real market data with some randomness
        base_price = 450
        hour_variation = np.sin(datetime.now().hour * np.pi / 12) * 5
        random_change = np.random.normal(0, 2)
        
        current_price = base_price + hour_variation + random_change
        change = random_change + hour_variation * 0.5
        
        return {
            'price': current_price,
            'change': change,
            'change_percent': f"{change/current_price*100:.2f}%"
        }
    
    def generate_forecast_prophet(self, df, periods=3):
        """Generate forecasts using Facebook Prophet if available"""
        if not self.prophet_available:
            return self.generate_forecast_linear(df, periods)
            
        try:
            prophet_df = df.rename(columns={'date': 'ds', 'value': 'y'})
            prophet_df = prophet_df[['ds', 'y']].dropna()
            
            if len(prophet_df) < 10:
                return None
                
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(prophet_df)
            
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
            return forecast
            
        except Exception as e:
            st.warning(f"Prophet forecast failed, using linear regression: {e}")
            return self.generate_forecast_linear(df, periods)
    
    def generate_forecast_linear(self, df, periods=3):
        """Generate forecasts using linear regression"""
        try:
            df = df.sort_values('date').reset_index(drop=True)
            df['time_index'] = range(len(df))
            
            X = df[['time_index']].values
            y = df['value'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            last_index = df['time_index'].max()
            future_indices = np.array([[last_index + i + 1] for i in range(periods)])
            future_values = model.predict(future_indices)
            
            # Create forecast dataframe
            last_date = df['date'].max()
            forecast_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
            
            forecast_df = pd.DataFrame({
                'date': forecast_dates,
                'value': future_values,
                'type': 'forecast'
            })
            
            return forecast_df
            
        except Exception as e:
            st.error(f"Linear forecast failed: {e}")
            return None
    
    def create_real_time_metrics(self):
        """Display real-time economic metrics"""
        st.subheader("📊 Real-Time Economic Indicators")
        
        market_data = self.get_live_market_data()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "S&P 500 ETF (Simulated)",
                f"${market_data['price']:.2f}",
                f"{market_data['change']:.2f} ({market_data['change_percent']})"
            )
        
        with col2:
            current_growth = np.random.normal(0.025, 0.005)
            st.metric(
                "Estimated GDP Growth",
                f"{current_growth:.2%}",
                f"{(current_growth - 0.02):.2%}"
            )
        
        with col3:
            current_inflation = np.random.normal(0.032, 0.002)
            st.metric(
                "Estimated Inflation",
                f"{current_inflation:.2%}",
                delta_color="inverse"
            )
        
        with col4:
            current_unemployment = np.random.normal(0.037, 0.001)
            st.metric(
                "Unemployment Rate",
                f"{current_unemployment:.2%}",
                f"{(0.038 - current_unemployment):.2%}"
            )
    
    def create_forecasting_section(self):
        """Create forecasting visualizations"""
        st.subheader("🔮 Economic Forecasting")
        
        fred_series = {
            'GDP': 'GDP',
            'Unemployment Rate': 'UNRATE', 
            'Inflation (CPI)': 'CPIAUCSL',
            'Industrial Production': 'INDPRO'
        }
        
        selected_series = st.selectbox("Select Economic Indicator", list(fred_series.keys()))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if self.prophet_available:
                forecast_method = st.radio(
                    "Forecasting Method",
                    ["Linear Regression", "Prophet (Facebook)"],
                    horizontal=True
                )
            else:
                forecast_method = "Linear Regression"
                st.info("Using Linear Regression (Prophet not available)")
        
        with col2:
            forecast_periods = st.slider("Forecast Periods (Months)", 1, 12, 6)
        
        if st.button("Generate Forecast"):
            with st.spinner("Generating forecast..."):
                series_id = fred_series[selected_series]
                historical_data = self.fetch_fred_data(series_id, "USA")
                
                if not historical_data.empty:
                    if forecast_method == "Prophet (Facebook)":
                        forecast = self.generate_forecast_prophet(historical_data, forecast_periods)
                    else:
                        forecast = self.generate_forecast_linear(historical_data, forecast_periods)
                    
                    self.plot_forecast(historical_data, forecast, selected_series, forecast_method)
    
    def plot_forecast(self, historical_data, forecast, selected_series, method):
        """Plot historical data and forecast"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_data['date'],
            y=historical_data['value'],
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast data
        if forecast is not None:
            if method == "Prophet (Facebook)":
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
                # Confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    fill=None,
                    mode='lines',
                    line=dict(color='red', width=0),
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    fill='tonexty',
                    mode='lines',
                    line=dict(color='red', width=0),
                    name='Uncertainty'
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=forecast['date'],
                    y=forecast['value'],
                    name='Forecast',
                    line=dict(color='red', width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f"{selected_series} - Historical Data & Forecast",
            xaxis_title="Date",
            yaxis_title=selected_series,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast summary
        if forecast is not None:
            self.show_forecast_summary(historical_data, forecast, method)
    
    def show_forecast_summary(self, historical_data, forecast, method):
        """Display forecast summary metrics"""
        st.subheader("📈 Forecast Summary")
        
        col1, col2, col3 = st.columns(3)
        
        last_historical = historical_data['value'].iloc[-1]
        
        with col1:
            if method == "Prophet (Facebook)":
                next_forecast = forecast['yhat'].iloc[-len(forecast)//4]  # First forecast point
            else:
                next_forecast = forecast['value'].iloc[0]
            
            change = ((next_forecast - last_historical) / last_historical) * 100
            st.metric(
                "Next Period Forecast",
                f"{next_forecast:.2f}",
                f"{change:.2f}%"
            )
        
        with col2:
            if method == "Prophet (Facebook)":
                final_forecast = forecast['yhat'].iloc[-1]
            else:
                final_forecast = forecast['value'].iloc[-1]
            
            total_change = ((final_forecast - last_historical) / last_historical) * 100
            st.metric(
                "Final Forecast",
                f"{final_forecast:.2f}",
                f"{total_change:.2f}%"
            )
        
        with col3:
            volatility = historical_data['value'].pct_change().std() * 100
            st.metric(
                "Historical Volatility",
                f"{volatility:.2f}%"
            )
    
    def create_world_bank_section(self):
        """Create section with World Bank data"""
        st.subheader("🌐 Global Economic Data")
        
        world_bank_indicators = {
            "GDP (current US$)": "NY.GDP.MKTP.CD",
            "GDP growth (annual %)": "NY.GDP.MKTP.KD.ZG",
            "Inflation, consumer prices (annual %)": "FP.CPI.TOTL.ZG",
            "Unemployment (% of labor force)": "SL.UEM.TOTL.ZS"
        }
        
        selected_indicator = st.selectbox("Select Indicator", list(world_bank_indicators.keys()))
        
        country_options = ["USA", "CAN", "GBR", "DEU", "FRA", "JPN", "AUS", "BRA", "IND", "CHN"]
        selected_countries = st.multiselect(
            "Select Countries",
            country_options,
            default=["USA", "CHN", "IND", "DEU"]
        )
        
        if selected_countries and st.button("Load Data"):
            with st.spinner("Fetching data..."):
                country_codes = ";".join(selected_countries)
                indicator_code = world_bank_indicators[selected_indicator]
                
                df = self.fetch_world_bank_data(indicator_code, country_codes)
                
                if not df.empty:
                    self.plot_world_bank_data(df, selected_indicator)
    
    def plot_world_bank_data(self, df, indicator):
        """Plot World Bank data"""
        pivot_df = df.pivot(index='year', columns='country', values='value')
        
        fig = px.line(
            pivot_df,
            title=f"{indicator} - World Bank Data",
            labels={'value': indicator, 'year': 'Year'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Latest data
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].sort_values('value', ascending=False)
        
        st.subheader(f"Latest Data ({latest_year})")
        st.dataframe(
            latest_data[['country', 'value']].rename(columns={'value': indicator}),
            use_container_width=True
        )
    
    def create_gis_map(self):
        """Create GIS map with economic data"""
        st.subheader("🗺️ Economic Indicators Map")
        
        sample_data = self.get_sample_economic_data()
        latest_data = sample_data[sample_data['date'] == sample_data['date'].max()]
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            latest_data,
            get_position=['lon', 'lat'],
            get_radius=500000,
            get_fill_color=[255, 0, 0, 160],
            get_line_color=[0, 0, 0],
            pickable=True,
            auto_highlight=True,
            radius_scale=100,
            radius_min_pixels=5,
            radius_max_pixels=50,
        )
        
        view_state = pdk.ViewState(
            latitude=20,
            longitude=0,
            zoom=1,
            pitch=0,
        )
        
        tooltip = {
            "html": "<b>{country}</b><br/>GDP: ${gdp:.2f}B<br/>Inflation: {inflation:.2%}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }
        
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip
        ))
    
    def run_dashboard(self):
        """Main dashboard execution"""
        st.title("🌍 Real-Time Economic Intelligence Dashboard")
        
        st.info("🔍 **Note**: Using simulated data. For real data, deploy to Streamlit Cloud with API keys.")
        
        # Real-time metrics
        self.create_real_time_metrics()
        
        # Main content
        tab1, tab2, tab3 = st.tabs(["📊 Forecasting", "🌐 Global Data", "🗺️ GIS Map"])
        
        with tab1:
            self.create_forecasting_section()
        
        with tab2:
            self.create_world_bank_section()
        
        with tab3:
            self.create_gis_map()

# Run the dashboard
if __name__ == "__main__":
    dashboard = RealTimeEconomicDashboard()
    dashboard.run_dashboard()
