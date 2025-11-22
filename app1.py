import streamlit as st
from streamlit_float import float_init, float_css_helper
from streamlit_option_menu import option_menu
from streamlit_toggle import toggle
from streamlit_searchbox import st_searchbox
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objs as go
from datetime import date, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from io import BytesIO
import xgboost as xgb
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="StockSense", layout="wide")






# --- Helpers ---
@st.cache_data(ttl=3600, show_spinner="Fetching stock data...")
def fetch_data(ticker, start, end, interval="1d"):
    """
    Fetch stock data from Yahoo Finance between given start and end dates.
    - Caches results for one hour to reduce redundant API calls.
    - Handles potential multi-index columns returned by yfinance.
    - Optimized for better performance and error handling.
    """
    try:
        # Track data requests
        if 'usage_stats' in st.session_state:
            st.session_state.usage_stats['data_requests'] += 1
        
        # Use yf.Ticker for more reliable data fetching
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end, interval=interval, auto_adjust=True, prepost=False)
        
        if df.empty:
            if 'usage_stats' in st.session_state:
                st.session_state.usage_stats['errors_encountered'] += 1
            return pd.DataFrame()
            
        # Clean up column names and reset index
        df.columns = df.columns.str.replace(' ', '_')
        df = df.reset_index()
        
        # Remove any rows with NaN values in critical columns
        critical_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [col for col in critical_cols if col in df.columns]
        df = df.dropna(subset=available_cols)
        
        return df.copy()
    except Exception as e:
        if 'usage_stats' in st.session_state:
            st.session_state.usage_stats['errors_encountered'] += 1
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner="Computing technical indicators...")
def add_technical_indicators_cached(df):
    """Cached version of technical indicators computation"""
    return add_technical_indicators(df)






def add_technical_indicators(df):
    """
    Adds multiple technical indicators to the stock DataFrame:
      - 7-day and 30-day Moving Averages
      - Bollinger Bands (Upper & Lower)
      - RSI (Relative Strength Index)
      - MACD (Moving Average Convergence Divergence) and Signal line
    """
    df = df.copy()

    # Moving averages (trend indicators)
    df['7_MA'] = df['Close'].rolling(7).mean()
    df['30_MA'] = df['Close'].rolling(30).mean()

    # Bollinger Bands for volatility
    df['STD20'] = df['Close'].rolling(20).std()
    df['UpperBB'] = df['30_MA'] + 2 * df['STD20']
    df['LowerBB'] = df['30_MA'] - 2 * df['STD20']

    # RSI (Relative Strength Index) for momentum
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(span=14).mean()
    roll_down = down.ewm(span=14).mean()
    RS = roll_up / roll_down
    df['RSI'] = 100 - (100 / (1 + RS))

    # MACD and Signal Line for trend and momentum shifts
    exp12 = df['Close'].ewm(span=12).mean()
    exp26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = exp12 - exp26
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    return df





def plot_candlestick_with_indicators(df, ticker):
    """
    Creates a modern candlestick chart with technical indicators:
      - 7-day MA, 30-day MA
      - Upper and Lower Bollinger Bands with fill
      - Professional styling and colors
    """
    fig = go.Figure()

    # Base candlestick chart with modern colors
    fig.add_trace(go.Candlestick(
        x=df['Date'], 
        open=df['Open'], 
        high=df['High'],
        low=df['Low'], 
        close=df['Close'], 
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='rgba(38, 166, 154, 0.1)',
        decreasing_fillcolor='rgba(239, 83, 80, 0.1)'
    ))

    # Add technical overlays with better styling
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['7_MA'], 
        mode='lines', 
        name='7 MA',
        line=dict(color='#ff9800', width=2),
        opacity=0.8
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['30_MA'], 
        mode='lines', 
        name='30 MA',
        line=dict(color='#2196f3', width=2),
        opacity=0.8
    ))
    
    # Bollinger Bands with fill
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['UpperBB'], 
        mode='lines', 
        name='Upper BB', 
        line=dict(dash='dot', color='#9c27b0', width=1),
        opacity=0.7,
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['LowerBB'], 
        mode='lines', 
        name='Lower BB', 
        line=dict(dash='dot', color='#9c27b0', width=1),
        opacity=0.7,
        fill='tonexty',
        fillcolor='rgba(156, 39, 176, 0.1)',
        showlegend=True
    ))

    # Configure layout with modern styling
    fig.update_layout(
        title=f"{ticker} Price with Technical Indicators",
        xaxis_rangeslider_visible=False,
        legend_title="Indicators",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        hovermode='x unified',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    return fig












def compute_best_trade(df):
    min_price = float('inf'); max_profit = 0.0; buy_date=None; sell_date=None; temp_buy=None
    for _, row in df.iterrows():
        p = row['Close']; d = row['Date']
        if p < min_price:
            min_price = p; temp_buy = d
        profit = p - min_price
        if profit > max_profit and temp_buy < d:
            max_profit = profit; buy_date = temp_buy; sell_date = d
    return buy_date, sell_date, min_price, max_profit






# --- Utility function to create sliding window features ---
def make_features(series, window_size=30):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)






def predict(df, days, model_type="LSTM"):
    """Safer predict: trains or loads cached models, returns future list, metrics dict, or error string."""
    try:
        series = df['Close'].values.astype(float)
    except Exception as e:
        return None, None, f"Data error: {e}"

    if len(series) < 60:
        return None, None, "Not enough data to train (need >= 60 data points)."

    X, y = make_features(series)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        if model_type == "LSTM":
            # reshape for LSTM
            X_train_l = np.expand_dims(X_train, axis=-1)
            X_test_l = np.expand_dims(X_test, axis=-1)

            model = load_keras_model_if_exists("lstm_model")
            if model is None:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(X_train_l.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(32),
                    Dense(16, activation='relu'),
                    Dense(1)
                ])
                model.compile(optimizer='adam', loss='mae')
                # use fewer epochs for demo stability
                model.fit(X_train_l, y_train, epochs=8, batch_size=16, verbose=0)
                try:
                    save_keras_model(model, "lstm_model")
                except Exception:
                    pass

            y_pred = model.predict(X_test_l).flatten()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # generate future predictions using rolling window
            seq = series[-30:].tolist()
            future = []
            for _ in range(days):
                x = np.array(seq[-30:]).reshape(1, 30, 1)
                next_val = model.predict(x, verbose=0)[0][0]
                seq.append(next_val)
                future.append(float(next_val))
            return future, {'r2': r2, 'mae': mae}, None

        else:
            # classical models
            if model_type == "LinearRegression":
                model = LinearRegression()
            elif model_type == "RandomForest":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif model_type == "XGBoost":
                model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
            else:
                return None, None, f"Unknown model type: {model_type}"

            saved = load_sklearn_model(model_type.lower())
            if saved is not None:
                model = saved
            else:
                model.fit(X_train, y_train)
                try:
                    save_sklearn_model(model, model_type.lower())
                except Exception:
                    pass

            y_pred = model.predict(X_test).flatten()
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)

            # predict future using last window
            seq = series[-30:].tolist()
            future = []
            for _ in range(days):
                x = np.array(seq[-30:]).reshape(1, -1)
                next_val = model.predict(x)[0]
                seq.append(next_val)
                future.append(float(next_val))
            return future, {'r2': r2, 'mae': mae}, None

    except Exception as e:
        return None, None, f"Model training/prediction failed: {e}"


def simulate_strategy(df, buy_date, sell_date):
    # simple buy-hold simulation between dates; returns fraction return or 0.0 on error
    if buy_date is None or sell_date is None:
        return 0.0
    try:
        row_buy = df.loc[df['Date'] == buy_date]
        row_sell = df.loc[df['Date'] == sell_date]
        if row_buy.empty or row_sell.empty:
            return 0.0
        buy_price = float(row_buy['Close'].iloc[0])
        sell_price = float(row_sell['Close'].iloc[0])
        return (sell_price - buy_price) / buy_price
    except Exception:
        return 0.0

def to_excel_bytes(df):
    from io import BytesIO
    import pandas as pd
    import numpy as np

    output = BytesIO()

    # Convert unsupported objects to strings
    df = df.applymap(lambda x: str(x) if isinstance(x, (list, dict, set)) else x)

    # Convert timezone-aware datetimes to naive
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)

    # Handle any object columns that may contain datetimes
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: x.tz_localize(None) if hasattr(x, 'tz_localize') else x
            )

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='data')
    return output.getvalue()







def create_metric_columns(data_dict, num_cols=4):
    """Create metric columns with consistent styling"""
    cols = st.columns(num_cols)
    for i, (key, value) in enumerate(data_dict.items()):
        with cols[i % num_cols]:
            st.metric(key, value)
    return cols


def create_download_buttons(df, ticker):
    """Create download buttons for data export"""
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False).encode()
        st.download_button(
            f"📊 Download {ticker} CSV", 
            data=csv, 
            file_name=f"{ticker}_data.csv", 
            mime="text/csv"
        )
    with col2:
        st.download_button(
            f"📈 Download {ticker} Excel", 
            data=to_excel_bytes(df), 
            file_name=f"{ticker}_data.xlsx"
        )
    with col3:
        # JSON export
        json_data = df.to_json(orient='records', date_format='iso')
        st.download_button(
            f"📋 Download {ticker} JSON", 
            data=json_data, 
            file_name=f"{ticker}_data.json", 
            mime="application/json"
        )


def create_advanced_export(df, ticker, analysis_type="data"):
    """Create advanced export options with multiple formats"""
    st.subheader("💾 Advanced Export Options")
    
    # Export format selection
    export_format = st.selectbox(
        "Select Export Format", 
        ["CSV", "Excel", "JSON", "All Formats"],
        help="Choose the format for your data export"
    )
    
    # Date range selection for export
    if 'Date' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            start_export = st.date_input("Export From", value=df['Date'].min().date(), key=f"export_start_{ticker}")
        with col2:
            end_export = st.date_input("Export To", value=df['Date'].max().date(), key=f"export_end_{ticker}")
        
        # Filter data by date range
        df_export = df[(df['Date'].dt.date >= start_export) & (df['Date'].dt.date <= end_export)]
    else:
        df_export = df
    
    # Column selection
    if len(df_export.columns) > 2:
        selected_columns = st.multiselect(
            "Select Columns to Export",
            options=df_export.columns.tolist(),
            default=df_export.columns.tolist(),
            help="Choose which columns to include in the export"
        )
        df_export = df_export[selected_columns]
    
    # Export buttons
    if export_format in ["CSV", "All Formats"]:
        csv_data = df_export.to_csv(index=False).encode()
        st.download_button(
            f"📊 Download {ticker} CSV", 
            data=csv_data, 
            file_name=f"{ticker}_{analysis_type}_{start_export}_to_{end_export}.csv", 
            mime="text/csv"
        )
    
    if export_format in ["Excel", "All Formats"]:
        st.download_button(
            f"📈 Download {ticker} Excel", 
            data=to_excel_bytes(df_export), 
            file_name=f"{ticker}_{analysis_type}_{start_export}_to_{end_export}.xlsx"
        )
    
    if export_format in ["JSON", "All Formats"]:
        json_data = df_export.to_json(orient='records', date_format='iso')
        st.download_button(
            f"📋 Download {ticker} JSON", 
            data=json_data, 
            file_name=f"{ticker}_{analysis_type}_{start_export}_to_{end_export}.json", 
            mime="application/json"
        )
    
    # Summary statistics
    st.subheader("📊 Export Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Records", len(df_export))
    with col2:
        st.metric("Columns", len(df_export.columns))
    with col3:
        st.metric("Date Range", f"{(end_export - start_export).days} days")
    with col4:
        st.metric("File Size", f"~{len(str(df_export)) // 1000} KB")


def display_stock_metrics(df, ticker):
    """Display key stock metrics in a clean format"""
    latest_close = df['Close'].iloc[-1]
    ma_30 = df['30_MA'].iloc[-1] if '30_MA' in df.columns else None
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
    
    metrics = {
        "Latest Close": f"${latest_close:.2f}",
        "30-day MA": f"${ma_30:.2f}" if ma_30 else "N/A",
        "RSI": f"{rsi:.2f}" if rsi else "N/A"
    }
    
    create_metric_columns(metrics, 3)


def get_market_overview():
    """Get market overview data for major indices"""
    try:
        indices = {
            'S&P 500': '^GSPC',
            'NASDAQ': '^IXIC', 
            'DOW': '^DJI',
            'VIX': '^VIX'
        }
        
        overview_data = {}
        for name, ticker in indices.items():
            try:
                df = fetch_data(ticker, date.today() - timedelta(days=5), date.today())
                if not df.empty:
                    current = df['Close'].iloc[-1]
                    previous = df['Close'].iloc[-2] if len(df) > 1 else current
                    change = ((current - previous) / previous) * 100
                    overview_data[name] = {
                        'price': current,
                        'change': change,
                        'ticker': ticker
                    }
            except Exception as e:
                continue
        return overview_data
    except Exception as e:
        return {}


def validate_ticker(ticker):
    """Validate if ticker symbol is valid"""
    if not ticker or len(ticker) < 1:
        return False, "Ticker symbol cannot be empty"
    
    # Basic validation - ticker should be alphanumeric and reasonable length
    if not ticker.replace('-', '').replace('.', '').isalnum():
        return False, "Ticker symbol contains invalid characters"
    
    if len(ticker) > 10:
        return False, "Ticker symbol is too long"
    
    return True, "Valid ticker"


def calculate_risk_metrics(df, ticker):
    """Calculate advanced risk metrics for a stock"""
    if df.empty or len(df) < 30:
        return {}
    
    returns = df['Close'].pct_change().dropna()
    
    # Basic risk metrics
    volatility = returns.std() * np.sqrt(252)  # Annualized volatility
    sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Value at Risk (VaR) - 95% confidence
    var_95 = np.percentile(returns, 5) * 100
    
    # Maximum Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # Beta calculation (requires market data)
    try:
        market_df = fetch_data('^GSPC', df['Date'].min(), df['Date'].max())
        if not market_df.empty and len(market_df) > 30:
            market_returns = market_df['Close'].pct_change().dropna()
            # Align dates
            common_dates = returns.index.intersection(market_returns.index)
            if len(common_dates) > 10:
                stock_returns_aligned = returns.loc[common_dates]
                market_returns_aligned = market_returns.loc[common_dates]
                covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0, 1]
                market_variance = np.var(market_returns_aligned)
                beta = covariance / market_variance if market_variance > 0 else 0
            else:
                beta = 0
        else:
            beta = 0
    except Exception as e:
        beta = 0
    
    return {
        'Volatility (Annualized)': f"{volatility * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe_ratio:.3f}",
        'VaR (95%)': f"{var_95:.2f}%",
        'Max Drawdown': f"{max_drawdown:.2f}%",
        'Beta': f"{beta:.3f}"
    }


def create_analytics_dashboard():
    """Create an analytics dashboard with usage statistics"""
    st.subheader("📊 Analytics Dashboard")
    
    # Usage statistics
    if 'usage_stats' not in st.session_state:
        st.session_state.usage_stats = {
            'tickers_analyzed': set(),
            'predictions_made': 0,
            'portfolio_updates': 0,
            'comparisons_made': 0,
            'session_start': date.today(),
            'data_requests': 0,
            'errors_encountered': 0
        }
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tickers Analyzed", len(st.session_state.usage_stats['tickers_analyzed']))
    with col2:
        st.metric("Predictions Made", st.session_state.usage_stats['predictions_made'])
    with col3:
        st.metric("Portfolio Updates", st.session_state.usage_stats['portfolio_updates'])
    with col4:
        st.metric("Comparisons Made", st.session_state.usage_stats['comparisons_made'])
    
    # Performance metrics
    st.subheader("⚡ Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Requests", st.session_state.usage_stats['data_requests'])
    with col2:
        st.metric("Errors Encountered", st.session_state.usage_stats['errors_encountered'])
    with col3:
        session_days = (date.today() - st.session_state.usage_stats['session_start']).days + 1
        st.metric("Session Duration", f"{session_days} days")
    
    # Most analyzed tickers
    if st.session_state.usage_stats['tickers_analyzed']:
        st.write("**Most Analyzed Tickers:**")
        ticker_list = list(st.session_state.usage_stats['tickers_analyzed'])
        for ticker in ticker_list[:5]:
            st.write(f"• {ticker}")


def create_user_preferences():
    """Create user preferences and settings management"""
    st.subheader("⚙️ User Preferences")
    
    # Initialize preferences
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'default_timeframe': 365,
            'default_interval': '1d',
            'auto_refresh': False,
            'chart_style': 'modern',
            'notifications': True,
            'data_cache_hours': 1
        }
    
    # Timeframe preferences
    st.write("**Default Settings:**")
    col1, col2 = st.columns(2)
    
    with col1:
        default_days = st.selectbox(
            "Default Timeframe",
            [30, 90, 180, 365, 730],
            index=[30, 90, 180, 365, 730].index(st.session_state.user_preferences['default_timeframe']),
            help="Default number of days for data analysis"
        )
        st.session_state.user_preferences['default_timeframe'] = default_days
    
    with col2:
        default_interval = st.selectbox(
            "Default Interval",
            ['1d', '1wk', '1mo'],
            index=['1d', '1wk', '1mo'].index(st.session_state.user_preferences['default_interval']),
            help="Default data interval"
        )
        st.session_state.user_preferences['default_interval'] = default_interval
    
    # Advanced preferences
    st.write("**Advanced Settings:**")
    col1, col2 = st.columns(2)
    
    with col1:
        auto_refresh = st.checkbox(
            "Auto-refresh Data",
            value=st.session_state.user_preferences['auto_refresh'],
            help="Automatically refresh data every hour"
        )
        st.session_state.user_preferences['auto_refresh'] = auto_refresh
        
        notifications = st.checkbox(
            "Enable Notifications",
            value=st.session_state.user_preferences['notifications'],
            help="Show notifications for alerts and updates"
        )
        st.session_state.user_preferences['notifications'] = notifications
    
    with col2:
        chart_style = st.selectbox(
            "Chart Style",
            ['modern', 'classic', 'minimal'],
            index=['modern', 'classic', 'minimal'].index(st.session_state.user_preferences['chart_style']),
            help="Visual style for charts"
        )
        st.session_state.user_preferences['chart_style'] = chart_style
        
        cache_hours = st.slider(
            "Data Cache (hours)",
            min_value=0.5,
            max_value=24.0,
            value=1.0,
            step=0.5,
            help="How long to cache data (0.5–24 hours)"
        )

        st.session_state.user_preferences['data_cache_hours'] = cache_hours
    
    # Reset preferences
    if st.button("🔄 Reset to Defaults"):
        st.session_state.user_preferences = {
            'default_timeframe': 365,
            'default_interval': '1d',
            'auto_refresh': False,
            'chart_style': 'modern',
            'notifications': True,
            'data_cache_hours': 1
        }
        st.success("Preferences reset to defaults!")
        st.rerun()


def add_data_quality_checks(df, ticker):
    """Add data quality validation and warnings"""
    warnings = []
    
    # Check for missing data
    missing_data = df.isnull().sum().sum()
    if missing_data > 0:
        warnings.append(f"⚠️ {missing_data} missing data points detected")
    
    # Check for zero volume days
    zero_volume_days = (df['Volume'] == 0).sum()
    if zero_volume_days > 0:
        warnings.append(f"⚠️ {zero_volume_days} days with zero volume")
    
    # Check for price anomalies
    price_changes = df['Close'].pct_change().abs()
    extreme_changes = (price_changes > 0.2).sum()  # 20% daily change
    if extreme_changes > 0:
        warnings.append(f"⚠️ {extreme_changes} days with extreme price changes (>20%)")
    
    # Check data recency
    if 'Date' in df.columns:
        days_old = (date.today() - df['Date'].max().date()).days
        if days_old > 7:
            warnings.append(f"⚠️ Data is {days_old} days old")
    
    # Display warnings
    if warnings:
        st.write("**Data Quality Warnings:**")
        for warning in warnings:
            st.write(warning)
    
    return len(warnings) == 0


def create_alert_system():
    """Create a simple alert system for price thresholds"""
    st.subheader("🔔 Price Alerts")
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        alert_ticker = st.text_input("Ticker", placeholder="AAPL", key="alert_ticker").upper()
    with col2:
        alert_price = st.number_input("Alert Price ($)", value=100.0, min_value=0.01, step=0.01)
    with col3:
        alert_condition = st.selectbox("Condition", ["Above", "Below"])
    
    if st.button("➕ Add Alert"):
        if alert_ticker:
            is_valid, message = validate_ticker(alert_ticker)
            if is_valid:
                st.session_state.alerts.append({
                    'ticker': alert_ticker,
                    'price': alert_price,
                    'condition': alert_condition,
                    'created': date.today()
                })
                st.success(f"Alert added for {alert_ticker}")
            else:
                st.error(message)
        else:
            st.error("Please enter a ticker symbol")
    
    # Display active alerts
    if st.session_state.alerts:
        st.write("**Active Alerts:**")
        for i, alert in enumerate(st.session_state.alerts):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"{alert['ticker']} {alert['condition']} ${alert['price']:.2f}")
            with col2:
                st.write(f"Created: {alert['created']}")
            with col3:
                if st.button("Check", key=f"check_{i}"):
                    df = fetch_data(alert['ticker'], date.today() - timedelta(days=1), date.today())
                    if not df.empty:
                        current_price = df['Close'].iloc[-1]
                        if alert['condition'] == "Above" and current_price > alert['price']:
                            st.success(f"🚨 ALERT: {alert['ticker']} is ${current_price:.2f} (above ${alert['price']:.2f})")
                        elif alert['condition'] == "Below" and current_price < alert['price']:
                            st.success(f"🚨 ALERT: {alert['ticker']} is ${current_price:.2f} (below ${alert['price']:.2f})")
                        else:
                            st.info(f"No alert triggered. Current price: ${current_price:.2f}")
            with col4:
                if st.button("❌", key=f"remove_alert_{i}"):
                    st.session_state.alerts.pop(i)
                    st.rerun()

# --- UI ---
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style=  -webkit-background-clip: text; 
                -webkit-text-fill-color: transparent; 
                background-clip: text;
                font-size: 3rem;
                font-weight: 800;
                margin-bottom: 0.5rem;'>
        📊 StockSense
    </h1>
    <p style='color: #64748b; font-size: 1.2rem; margin: 0;'>
        Advanced Financial Dashboard & AI-Powered Analysis
    </p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🎛️ Controls")

# Market Overview
st.sidebar.subheader("📊 Market Overview")
if st.sidebar.button("🔄 Refresh Market Data"):
    with st.spinner("Loading market data..."):
        market_data = get_market_overview()
        if market_data:
            for name, data in market_data.items():
                change_color = "🟢" if data['change'] >= 0 else "🔴"
                st.sidebar.metric(
                    name, 
                    f"${data['price']:.2f}", 
                    f"{change_color} {data['change']:+.2f}%"
                )
        else:
            st.sidebar.error("Unable to load market data")

# Quick Market Status
market_data = get_market_overview()
if market_data:
    st.sidebar.write("**Quick Status:**")
    for name, data in market_data.items():
        change_color = "🟢" if data['change'] >= 0 else "🔴"
        st.sidebar.write(f"{change_color} {name}: {data['change']:+.2f}%")

st.sidebar.markdown("---")






# ===================
# THEME TOGGLE 
# ===================
# Initialize floating elements
float_init()

# --- Initialize theme state ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Analytics, Preferences, and Alert System in Sidebar
with st.sidebar:
    create_analytics_dashboard()
    st.markdown("---")
    create_user_preferences()
    st.markdown("---")
    create_alert_system()









# =============================================
# FLOATING SEARCH BAR (with Autocomplete)
# =============================================
from data import data

# === Model save/load helpers ===
import os
import joblib
from tensorflow.keras.models import load_model as keras_load_model

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def save_sklearn_model(model, model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    joblib.dump(model, path)
    return path

def load_sklearn_model(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            # failed to load, remove corrupted file
            try:
                os.remove(path)
            except Exception as e:
                pass
            return None
    return None

def save_keras_model(model, model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}_keras")
    model.save(path)
    return path

def load_keras_model_if_exists(model_name):
    path = os.path.join(MODEL_DIR, f"{model_name}_keras")
    if os.path.exists(path):
        try:
            return keras_load_model(path)
        except Exception:
            return None
    return None

# initialize usage stats in session state if using streamlit
try:
    import streamlit as _st
    if 'usage_stats' not in _st.session_state:
        _st.session_state['usage_stats'] = {'data_requests': 0, 'errors_encountered': 0}
except Exception:
    pass

# === End helpers ===

# --- Define the search function ---
def search_stock(query: str):
    """Return suggestions that match the query."""
    if not query:
        return []
    return [
        f"{item['ticker']} — {item['name']}"
        for item in data
        if query.lower() in item['ticker'].lower() or query.lower() in item['name'].lower()
    ]


# --- Searchbox widget with live suggestions ---
search_selection = st_searchbox(
    search_function=search_stock,
    key="stock_search",
    placeholder="🔍 Search for a stock (by ticker or name)...",
    default=None,
)


# --- If user selects something ---
if search_selection:
    selected_ticker = search_selection.split(" — ")[0]
    selected_stock = next((item for item in data if item['ticker'] == selected_ticker), None)

    if selected_stock:
        st.markdown(f"## 🏢 {selected_stock['name']} ({selected_stock['ticker']})")

        # --- Fetch data from yfinance ---
        stock = yf.Ticker(selected_ticker)
        info = stock.info
        hist = stock.history(period="6mo")

        # --- Display KPIs ---
        st.markdown("### 💡 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
        with col2:
            st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}" if info.get('marketCap') else "N/A")
        with col3:
            st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        with col4:
            st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

        # --- Show 6-month change ---
        if not hist.empty:
            last_close = hist["Close"].iloc[-1]
            first_close = hist["Close"].iloc[0]
            change_pct = ((last_close - first_close) / first_close) * 100
            st.metric("6-Month Change (%)", f"{change_pct:.2f}%")

        # --- Chart ---
        st.markdown("### 📉 Price History (Last 6 Months)")
        st.line_chart(hist["Close"])

        # --- Additional info ---
        with st.expander("Show Additional Info"):
            st.write({
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Country": info.get("country", "N/A"),
                "Website": info.get("website", "N/A")
            })

        # --- Raw historical data ---
        with st.expander("View Raw Historical Data"):
            st.dataframe(hist.tail(20))








tabs = st.tabs(["Overview","Technical Analysis","AI Predictions","Portfolio","Compare"])
with tabs[0]:
    st.header("📈 Overview")

    # Input controls
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Analysis Parameters")
        tickers_input = st.text_input("Enter tickers (comma separated)", value="AAPL,MSFT", placeholder="e.g., AAPL,MSFT,GOOGL")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        # Validate tickers
        invalid_tickers = []
        for ticker in tickers:
            is_valid, message = validate_ticker(ticker)
            if not is_valid:
                invalid_tickers.append(f"{ticker}: {message}")

        if invalid_tickers:
            st.error("Invalid ticker symbols:")
            for error in invalid_tickers:
                st.error(f"• {error}")

        col1a, col1b, col1c = st.columns(3)
        with col1a:
            start_date = st.date_input("Start date", value=date.today() - timedelta(days=365), key="overview_start")
        with col1b:
            end_date = st.date_input("End date", value=date.today(), key="overview_end")
        with col1c:
            interval = st.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
        run = st.button("🚀 Fetch & Analyze", type="primary")

    with col2:
        st.subheader("📋 Watchlist")
        if 'watchlist' not in st.session_state:
            st.session_state.watchlist = []

        new_save = st.text_input("Ticker to bookmark", placeholder="e.g., TSLA")
        if st.button("➕ Add to Watchlist"):
            if new_save and new_save.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_save.upper())
                st.success(f"Added {new_save.upper()} to watchlist!")

        if st.session_state.watchlist:
            st.write("**Saved Tickers:**")
            for t in st.session_state.watchlist:
                st.write(f"• {t}")
        else:
            st.info("No tickers in watchlist yet")

    # Analysis results
    if run:
        st.markdown("---")
        for t in tickers:
            with st.container():
                st.subheader(f"📊 {t} Analysis")

                # Fetch and process data
                df = fetch_data(t, start_date, end_date, interval)
                if df.empty:
                    st.error(f"❌ No data available for {t}")
                    continue

                df = add_technical_indicators_cached(df)

                # Track usage
                if 'usage_stats' not in st.session_state:
                    st.session_state.usage_stats = {
                        'tickers_analyzed': set(),
                        'predictions_made': 0,
                        'portfolio_updates': 0,
                        'comparisons_made': 0,
                        'session_start': date.today(),
                        'data_requests': 0,
                        'errors_encountered': 0
                    }
                st.session_state.usage_stats['tickers_analyzed'].add(t)

                # Display chart
                fig = plot_candlestick_with_indicators(df, t)
                st.plotly_chart(fig, use_container_width=True)

                # Trading opportunity analysis
                buy, sell, price, profit = compute_best_trade(df)
                if buy and sell:
                    st.success(f"💰 Best buy: `{buy.date()}` | Best sell: `{sell.date()}` | Potential profit: ${profit:.2f}")
                else:
                    st.info("ℹ️ No clear one-time profit opportunity in this period")

                # Key metrics
                st.subheader("📊 Key Metrics")
                display_stock_metrics(df, t)

                # Download options
                st.subheader("💾 Export Data")
                create_download_buttons(df, t)

                st.markdown("---")







with tabs[1]:
    st.header("🔍 Technical Analysis")

    # Input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        t_input = st.text_input("Ticker", value="AAPL", placeholder="e.g., AAPL")
        t = t_input.upper() if t_input else ""

        # Validate ticker
        if t:
            is_valid, message = validate_ticker(t)
            if not is_valid:
                st.error(f"Invalid ticker: {message}")
    with col2:
        start = st.date_input("Start Date", value=date.today() - timedelta(days=365), key="ta_start")
    with col3:
        end = st.date_input("End Date", value=date.today(), key="ta_end")

    if st.button("🔍 Analyze Technical Indicators", type="primary"):
        df = fetch_data(t, start, end)
        if df.empty:
            st.error("❌ No data found for this ticker and date range.")
        else:
            # Add technical indicators
            df = add_technical_indicators_cached(df)

            # Data quality checks
            st.subheader("🔍 Data Quality Check")
            data_quality_ok = add_data_quality_checks(df, t)

            if data_quality_ok:
                st.success("✅ Data quality checks passed - all data looks good!")

            # Main price chart with indicators
            st.subheader(f"📊 {t} Price Chart with Technical Indicators")
            st.plotly_chart(plot_candlestick_with_indicators(df, t), use_container_width=True)

            # Key metrics
            st.subheader("📈 Key Technical Metrics")
            display_stock_metrics(df, t)

            # RSI and MACD analysis
            st.subheader("📊 Momentum Indicators")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**RSI (Relative Strength Index)**")
                rsi_value = df['RSI'].iloc[-1]
                if rsi_value > 70:
                    st.warning(f"RSI: {rsi_value:.2f} - Overbought")
                elif rsi_value < 30:
                    st.success(f"RSI: {rsi_value:.2f} - Oversold")
                else:
                    st.info(f"RSI: {rsi_value:.2f} - Neutral")

            with col2:
                st.markdown("**MACD Signal**")
                macd = df['MACD'].iloc[-1]
                signal = df['Signal'].iloc[-1]
                if macd > signal:
                    st.success("MACD: Bullish Signal")
                else:
                    st.error("MACD: Bearish Signal")

            # RSI and MACD charts
            st.subheader("📈 RSI and MACD Trend Analysis")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI', line=dict(color='#ff6b6b', width=2)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD', line=dict(color='#4ecdc4', width=2)))
            fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal'], name='Signal', line=dict(color='#45b7d1', width=2)))

            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Overbought (70)")
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Oversold (30)")

            fig.update_layout(
                title="RSI & MACD Indicators",
                legend_title="Indicators",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Volume Analysis
            st.subheader("📊 Volume Analysis")
            col1, col2 = st.columns(2)

            with col1:
                fig_volume = go.Figure()
                fig_volume.add_trace(go.Bar(
                    x=df['Date'],
                    y=df['Volume'],
                    name='Volume',
                    marker_color='rgba(55, 128, 191, 0.7)'
                ))
                fig_volume.update_layout(
                    title="Trading Volume",
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    height=300
                )
                st.plotly_chart(fig_volume, use_container_width=True)

            with col2:
                avg_volume = df['Volume'].mean()
                recent_volume = df['Volume'].iloc[-5:].mean()
                volume_trend = "📈 Increasing" if recent_volume > avg_volume else "📉 Decreasing"
                st.metric("Average Volume", f"{avg_volume:,.0f}")
                st.metric("Recent Volume", f"{recent_volume:,.0f}")
                st.metric("Volume Trend", volume_trend)

            # Moving averages comparison
            st.subheader("📈 Price vs Moving Averages")
            st.line_chart(df.set_index('Date')[['Close', '7_MA', '30_MA']])

            # Candlestick Pattern Analysis
            st.subheader("🕯️ Candlestick Pattern Analysis")
            if len(df) >= 2:
                last_candle = df.iloc[-1]
                prev_candle = df.iloc[-2]

                body_size = abs(last_candle['Close'] - last_candle['Open'])
                total_range = last_candle['High'] - last_candle['Low']
                is_doji = body_size < (total_range * 0.1)

                lower_shadow = min(last_candle['Open'], last_candle['Close']) - last_candle['Low']
                upper_shadow = last_candle['High'] - max(last_candle['Open'], last_candle['Close'])
                is_hammer = lower_shadow > (body_size * 2) and upper_shadow < body_size

                is_bullish_engulfing = (
                    prev_candle['Close'] < prev_candle['Open'] and
                    last_candle['Close'] > last_candle['Open'] and
                    last_candle['Open'] < prev_candle['Close'] and
                    last_candle['Close'] > prev_candle['Open']
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success("🕯️ Doji Pattern - Indecision") if is_doji else st.info("No Doji pattern")
                with col2:
                    st.success("🔨 Hammer Pattern - Potential Reversal") if is_hammer else st.info("No Hammer pattern")
                with col3:
                    st.success("📈 Bullish Engulfing - Strong Buy Signal") if is_bullish_engulfing else st.info("No Engulfing pattern")

            # Risk Analysis
            st.subheader("⚠️ Risk Analysis")
            risk_metrics = calculate_risk_metrics(df, t)
            if risk_metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Volatility", risk_metrics.get('Volatility (Annualized)', 'N/A'))
                    st.metric("Sharpe Ratio", risk_metrics.get('Sharpe Ratio', 'N/A'))
                with col2:
                    st.metric("VaR (95%)", risk_metrics.get('VaR (95%)', 'N/A'))
                    st.metric("Max Drawdown", risk_metrics.get('Max Drawdown', 'N/A'))
                with col3:
                    st.metric("Beta", risk_metrics.get('Beta', 'N/A'))

                volatility = float(risk_metrics.get('Volatility (Annualized)', '0%').replace('%', ''))
                if volatility > 30:
                    st.warning("⚠️ High volatility stock - consider position sizing carefully")
                elif volatility < 15:
                    st.success("✅ Low volatility stock - relatively stable")
                else:
                    st.info("ℹ️ Moderate volatility stock")

            # Download data
            st.subheader("💾 Export Analysis Data")
            create_download_buttons(df, t)







with tabs[2]:
    st.header("🤖 AI Predictions")

    # Input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        t_input = st.text_input("Ticker for Prediction", value="AAPL", key="pred_ticker", placeholder="e.g., AAPL")
        t = t_input.upper() if t_input else ""

        # Validate ticker
        if t:
            is_valid, message = validate_ticker(t)
            if not is_valid:
                st.error(f"Invalid ticker: {message}")
    with col2:
        days = st.slider("Days to Predict", 1, 30, 7, help="Number of days into the future to predict")
    with col3:
        model_choice = st.selectbox(
            "AI Model",
            ["LinearRegression", "RandomForest", "XGBoost", "LSTM"],
            help="Choose the machine learning model for prediction"
        )

    model_info = {
        "LinearRegression": "📊 Simple linear regression - fast and interpretable",
        "RandomForest": "🌲 Ensemble method - good balance of accuracy and speed",
        "XGBoost": "⚡ Gradient boosting - high accuracy, slower training",
        "LSTM": "🧠 Deep learning - best for complex patterns, requires more data"
    }

    st.info(f"**Selected Model:** {model_info[model_choice]}")

    if st.button("🚀 Generate AI Prediction", type="primary"):
        with st.spinner("Fetching data and training model..."):
            df = fetch_data(t, date.today() - timedelta(days=800), date.today())

            if df.empty:
                st.error("❌ No data available for prediction. Please check the ticker symbol.")
            else:
                with st.spinner(f"Training {model_choice} model and generating predictions..."):
                    preds, metrics, err = predict(df, days, model_choice)

                    # Track usage
                    if 'usage_stats' not in st.session_state:
                        st.session_state.usage_stats = {
                            'tickers_analyzed': set(),
                            'predictions_made': 0,
                            'portfolio_updates': 0,
                            'comparisons_made': 0,
                            'session_start': date.today(),
                            'data_requests': 0,
                            'errors_encountered': 0
                        }
                    st.session_state.usage_stats['predictions_made'] += 1
                    st.session_state.usage_stats['tickers_analyzed'].add(t)

                if err:
                    st.error(f"❌ {err}")
                    st.info("💡 Try using a ticker with more historical data or a simpler model.")
                else:
                    base = df['Date'].iloc[-1]
                    future_dates = [base + timedelta(days=i + 1) for i in range(days)]
                    df_future = pd.DataFrame({'Date': future_dates, 'Predicted': preds})

                    # Summary
                    st.subheader("📈 Prediction Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        current_price = df['Close'].iloc[-1]
                        st.metric("Current Price", f"${current_price:.2f}")
                    with col2:
                        predicted_price = preds[-1]
                        change = ((predicted_price - current_price) / current_price) * 100
                        st.metric("Predicted Price", f"${predicted_price:.2f}", f"{change:+.2f}%")
                    with col3:
                        st.metric("Model Confidence (R²)", f"{metrics['r2']:.1%}")

                    # Chart
                    st.subheader("📊 Price Prediction Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df['Date'],
                        y=df['Close'],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=future_dates,
                        y=preds,
                        mode='lines+markers',
                        name='AI Prediction',
                        line=dict(color='#ef4444', width=2, dash='dot'),
                        marker=dict(size=6, color='#ef4444')
                    ))
                    fig.add_shape(
                        type="line",
                        x0=pd.to_datetime(base).to_pydatetime(),
                        x1=pd.to_datetime(base).to_pydatetime(),
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color="gray", width=1, dash="dash")
                    )
                    fig.add_annotation(
                        x=pd.to_datetime(base).to_pydatetime(),
                        y=1,
                        xref="x",
                        yref="paper",
                        text="Prediction Start",
                        showarrow=False,
                        yanchor="bottom"
                    )
                    fig.update_layout(
                        title=f"{t} Price Prediction ({model_choice})",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        legend_title="Data Type",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Model performance
                    st.subheader("📊 Model Performance")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("R² Score", f"{metrics['r2']:.3f}", help="Higher is better (1.0 = perfect prediction)")
                    with col2:
                        st.metric("Mean Absolute Error", f"${metrics['mae']:.2f}", help="Average prediction error in dollars")

                    # Table
                    st.subheader("📅 Detailed Predictions")
                    df_display = df_future.copy()
                    df_display['Date'] = df_display['Date'].dt.strftime('%Y-%m-%d')
                    df_display['Predicted'] = df_display['Predicted'].round(2)
                    df_display.columns = ['Date', 'Predicted Price ($)']
                    st.dataframe(df_display, use_container_width=True)

                    # Downloads
                    st.subheader("💾 Export Predictions")
                    create_download_buttons(df_future, f"{t}_predictions")








with tabs[3]:
    st.header("💼 Portfolio Management")

    # Initialize portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}

    # Portfolio controls
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("➕ Add to Portfolio")
        t_input = st.text_input("Ticker Symbol", placeholder="e.g., AAPL", key="portfolio_ticker")
        t = t_input.upper() if t_input else ""

        # Validate ticker
        if t:
            is_valid, message = validate_ticker(t)
            if not is_valid:
                st.error(f"Invalid ticker: {message}")

        qty = st.number_input("Quantity", value=1, min_value=1, step=1, help="Number of shares to add")

        col1a, col1b = st.columns(2)
        with col1a:
            if st.button("➕ Add to Portfolio", type="primary"):
                if t:
                    if t in st.session_state.portfolio:
                        st.session_state.portfolio[t] += qty
                        st.success(f"✅ Added {qty} shares of {t}. Total: {st.session_state.portfolio[t]} shares")
                    else:
                        st.session_state.portfolio[t] = qty
                        st.success(f"✅ Added {t} to portfolio with {qty} shares")

                    if 'usage_stats' not in st.session_state:
                        st.session_state.usage_stats = {
                            'tickers_analyzed': set(),
                            'predictions_made': 0,
                            'portfolio_updates': 0,
                            'comparisons_made': 0,
                            'session_start': date.today(),
                            'data_requests': 0,
                            'errors_encountered': 0
                        }

                    st.session_state.usage_stats['portfolio_updates'] += 1
                    st.session_state.usage_stats['tickers_analyzed'].add(t)
                else:
                    st.error("Please enter a ticker symbol")

        with col1b:
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio = {}
                st.success("Portfolio cleared!")

    with col2:
        st.subheader("📋 Current Holdings")
        if st.session_state.portfolio:
            for k, v in st.session_state.portfolio.items():
                col2a, col2b, col2c = st.columns([2, 1, 1])
                with col2a:
                    st.write(f"**{k}**")
                with col2b:
                    st.write(f"{v} shares")
                with col2c:
                    if st.button("❌", key=f"remove_{k}", help=f"Remove {k}"):
                        del st.session_state.portfolio[k]
                        st.rerun()
        else:
            st.info("No holdings in portfolio yet")

    # Portfolio valuation
    if st.button("💰 Refresh Portfolio Valuation", type="secondary"):
        if st.session_state.portfolio:
            with st.spinner("Fetching current prices..."):
                rows = []
                total_value = 0
                total_cost = 0

                for k, v in st.session_state.portfolio.items():
                    dfk = fetch_data(k, date.today() - timedelta(days=30), date.today())
                    if dfk.empty:
                        st.warning(f"Could not fetch data for {k}")
                        continue

                    current_price = dfk['Close'].iloc[-1]
                    cost_basis = current_price * 0.9  # assume cost basis is 10% lower
                    current_value = current_price * v
                    cost_value = cost_basis * v
                    gain_loss = current_value - cost_value
                    gain_loss_pct = (gain_loss / cost_value) * 100

                    rows.append({
                        'Ticker': k,
                        'Quantity': v,
                        'Current Price': f"${current_price:.2f}",
                        'Cost Basis': f"${cost_basis:.2f}",
                        'Current Value': f"${current_value:.2f}",
                        'Gain/Loss': f"${gain_loss:.2f}",
                        'Gain/Loss %': f"{gain_loss_pct:+.2f}%"
                    })

                    total_value += current_value
                    total_cost += cost_value

                if rows:
                    st.subheader("📊 Portfolio Valuation")
                    pf = pd.DataFrame(rows)
                    st.dataframe(pf, use_container_width=True)

                    # Portfolio summary
                    st.subheader("💰 Portfolio Summary")
                    col1, col2, col3, col4 = st.columns(4)

                    total_gain_loss = total_value - total_cost
                    total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0

                    with col1:
                        st.metric("Total Value", f"${total_value:.2f}")
                    with col2:
                        st.metric("Total Cost", f"${total_cost:.2f}")
                    with col3:
                        st.metric("Gain/Loss", f"${total_gain_loss:.2f}")
                    with col4:
                        st.metric("Gain/Loss %", f"{total_gain_loss_pct:+.2f}%")

                    # Pie chart
                    if len(rows) > 1:
                        st.subheader("📈 Portfolio Allocation")
                        fig = go.Figure(data=[go.Pie(
                            labels=[row['Ticker'] for row in rows],
                            values=[float(row['Current Value'].replace('$', '').replace(',', '')) for row in rows],
                            hole=0.3
                        )])
                        fig.update_layout(title="Portfolio Allocation by Value", showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("No valid data found for portfolio holdings")
        else:
            st.info("Add some stocks to your portfolio to see valuation")







with tabs[4]:
    st.header("📊 Stock Comparison")

    # Input controls
    col1, col2, col3 = st.columns(3)
    with col1:
        ticks_input = st.text_input(
            "Tickers to Compare",
            value="AAPL,MSFT,GOOGL",
            placeholder="e.g., AAPL,MSFT,GOOGL",
            help="Enter tickers separated by commas"
        )
        ticks = [t.strip().upper() for t in ticks_input.split(",") if t.strip()]

        # Validate tickers
        if ticks:
            invalid_tickers = []
            for ticker in ticks:
                is_valid, message = validate_ticker(ticker)
                if not is_valid:
                    invalid_tickers.append(f"{ticker}: {message}")

            if invalid_tickers:
                st.error("Invalid ticker symbols:")
                for error in invalid_tickers:
                    st.error(f"• {error}")

    with col2:
        start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365), key="compare_start")
    with col3:
        end_date = st.date_input("End Date", value=date.today(), key="compare_end")

    if st.button("🔍 Compare Stocks", type="primary"):
        if not ticks:
            st.error("Please enter at least one valid ticker.")
        else:
            with st.spinner("Fetching and comparing data..."):
                all_data = {}
                for t in ticks:
                    df = fetch_data(t, start_date, end_date)
                    if not df.empty:
                        df = add_technical_indicators_cached(df)
                        all_data[t] = df
                        if 'usage_stats' in st.session_state:
                            st.session_state.usage_stats['comparisons_made'] += 1
                            st.session_state.usage_stats['tickers_analyzed'].add(t)
                    else:
                        st.warning(f"⚠️ No data for {t}")

                if not all_data:
                    st.error("❌ No valid data to compare.")
                else:
                    # Combined closing price comparison
                    st.subheader("📈 Closing Price Comparison")
                    combined = pd.DataFrame({
                        t: df.set_index("Date")["Close"] for t, df in all_data.items()
                    })
                    st.line_chart(combined)

                    # Normalized performance comparison
                    st.subheader("📊 Normalized Performance (%)")
                    normalized = combined / combined.iloc[0] * 100
                    st.line_chart(normalized)

                    # Summary statistics
                    st.subheader("📋 Summary Statistics")
                    summary = pd.DataFrame({
                        t: {
                            "Mean": df["Close"].mean(),
                            "Std Dev": df["Close"].std(),
                            "Min": df["Close"].min(),
                            "Max": df["Close"].max(),
                            "Final Price": df["Close"].iloc[-1],
                            "Return (%)": ((df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1) * 100,
                        }
                        for t, df in all_data.items()
                    }).T

                    st.dataframe(summary.style.format({
                        "Mean": "{:.2f}",
                        "Std Dev": "{:.2f}",
                        "Min": "{:.2f}",
                        "Max": "{:.2f}",
                        "Final Price": "{:.2f}",
                        "Return (%)": "{:.2f}%"
                    }), use_container_width=True)

                    # Correlation heatmap
                    st.subheader("🔗 Correlation Matrix")
                    corr = combined.corr()
                    fig = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.columns,
                        colorscale='Blues'
                    ))
                    fig.update_layout(
                        title="Stock Return Correlation Matrix",
                        xaxis_title="Ticker",
                        yaxis_title="Ticker"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Export comparison data
                    st.subheader("💾 Export Comparison Data")
                    create_download_buttons(combined.reset_index(), "stock_comparison")

    st.markdown("---")
    st.info("💡 Tip: Use normalized performance to compare relative growth among stocks with different price levels.")