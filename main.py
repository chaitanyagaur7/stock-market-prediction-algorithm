# Enhanced app.py with improved UI and additional features

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from utils.data_loader import load_stock_data
from utils.visualizer import plot_stock_data, plot_predictions
from models.hybrid_model import train_models

# App configuration
st.set_page_config(
    page_title="Advanced Stock Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stock-info {
        font-size: 1.1rem;
        padding: 0.5rem;
        background-color: #e3f2fd;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-card {
        border-left: 4px solid #42a5f5;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    .footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.8rem;
        color: #9e9e9e;
    }
</style>
""", unsafe_allow_html=True)

# Cache popular stocks data
@st.cache_data(ttl=3600)
def get_stock_list():
    # List of popular stocks with their tickers and sectors
    popular_stocks = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'INTC', 'AMD', 'ORCL'],
        'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK'],
        'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'ABT', 'UNH', 'LLY', 'TMO', 'DHR', 'BMY'],
        'Consumer': ['KO', 'PEP', 'PG', 'WMT', 'COST', 'MCD', 'NKE', 'SBUX', 'DIS', 'HD'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'PSX', 'BP', 'VLO', 'MPC']
    }
    
    # Create a flat list for search
    all_stocks = []
    for sector, tickers in popular_stocks.items():
        for ticker in tickers:
            try:
                # Get company name
                info = yf.Ticker(ticker).info
                name = info.get('shortName', ticker)
                all_stocks.append({"ticker": ticker, "name": name, "sector": sector})
            except:
                all_stocks.append({"ticker": ticker, "name": ticker, "sector": sector})
    
    return all_stocks, popular_stocks

# Fetch stock data with progress indication
def fetch_stock_with_progress(ticker):
    with st.spinner(f'Fetching data for {ticker}...'):
        data = load_stock_data(ticker)
        return data

# Function to create future prediction dates
def get_future_dates(last_date, days=5):
    future_dates = []
    current_date = last_date
    for _ in range(days):
        next_date = current_date + timedelta(days=1)
        # Skip weekends
        while next_date.weekday() > 4:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        future_dates.append(next_date)
        current_date = next_date
    return future_dates

# Function to predict future prices
def predict_future(model, last_data, days=5):
    # This is a simple placeholder for future predictions
    # In a real implementation, this would use the trained model to generate predictions
    last_price = last_data['Close'].iloc[-1]
    future_prices = []
    current_price = last_price
    
    # Generate some simulated future predictions
    for _ in range(days):
        # Simulate a random walk with slight upward bias based on recent trend
        recent_trend = last_data['Close'].pct_change().mean() * 100
        change = np.random.normal(recent_trend, 0.7)  # Adjust volatility as needed
        current_price = current_price * (1 + change/100)
        future_prices.append(current_price)
    
    return future_prices

# Create an interactive stock chart with technical indicators
def create_interactive_chart(data, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.1, subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                       row_heights=[0.7, 0.3])
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add moving averages
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['MA50'],
            line=dict(color='blue', width=1.5),
            name="50-day MA"
        ),
        row=1, col=1
    )
    
    # Add volume bar chart
    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data['Volume'],
            name="Volume",
            marker_color='rgba(0, 150, 255, 0.6)'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Interactive Stock Analysis",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

# Main app
def main():
    st.markdown(
        """
        <style>
        .navbar {
            background-color: #4CAF50;
            padding: 1rem;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
            border-radius: 0px 0px 10px 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown('<div class="main-header">Hybrid Stock Price Prediction App</div>', unsafe_allow_html=True)
    
    # Get stock lists
    all_stocks, popular_stocks = get_stock_list()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Stock Analysis", "Market Overview"])
    
    # Stock selection section with search and categories
    st.sidebar.markdown("## Stock Selection")
    
    # Search for stocks
    search_placeholder = st.sidebar.empty()
    search_term = search_placeholder.text_input("Search stocks (e.g., AAPL, Microsoft)", key="search")
    
    # Filter stocks based on search
    filtered_stocks = []
    if search_term:
        search_term = search_term.upper()
        filtered_stocks = [s for s in all_stocks if search_term in s["ticker"] or search_term.lower() in s["name"].lower()]
        
        if filtered_stocks:
            selected_stock = st.sidebar.selectbox(
                "Select from results:",
                options=filtered_stocks,
                format_func=lambda x: f"{x['ticker']} - {x['name']} ({x['sector']})"
            )
            ticker = selected_stock["ticker"]
        else:
            st.sidebar.warning("No matching stocks found.")
            ticker = "MSFT"  # Default
    else:
        # Show categories if no search
        sector = st.sidebar.selectbox("Select sector:", list(popular_stocks.keys()))
        ticker = st.sidebar.selectbox("Select stock:", popular_stocks[sector])
    
    # Time period selection
    period = st.sidebar.select_slider(
        "Select time period:",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"],
        value="1y"
    )
    
    # Analysis options
    st.sidebar.markdown("## Analysis Options")
    show_technicals = st.sidebar.checkbox("Show Technical Indicators", value=True)
    show_predictions = st.sidebar.checkbox("Show Predictions", value=True)
    # Fetch data
    data = fetch_stock_with_progress(ticker)
    
    if page == "Stock Analysis":
        # Stock info header
        try:
            stock_info = yf.Ticker(ticker).info
            company_name = stock_info.get('shortName', ticker)
            current_price = stock_info.get('currentPrice', data['Close'].iloc[-1])
            prev_close = stock_info.get('previousClose', data['Close'].iloc[-2])
            price_change = current_price - prev_close
            price_change_pct = (price_change / prev_close) * 100
            
            # Display stock info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label=f"{company_name} ({ticker})",
                    value=f"${current_price:.2f}",
                    delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
                )
            with col2:
                st.metric(
                    label="52-Week Range",
                    value=f"${stock_info.get('fiftyTwoWeekLow', 'N/A'):.2f} - ${stock_info.get('fiftyTwoWeekHigh', 'N/A'):.2f}"
                )
            with col3:
                st.metric(
                    label="Volume",
                    value=f"{stock_info.get('volume', 'N/A'):,}"
                )
        except Exception as e:
            st.warning(f"Could not retrieve detailed info for {ticker}: {e}")
            st.subheader(f"Analysis for {ticker}")
        
        # Interactive chart
        st.markdown('<div class="sub-header">Interactive Stock Chart</div>', unsafe_allow_html=True)
        interactive_chart = create_interactive_chart(data, ticker)
        st.plotly_chart(interactive_chart, use_container_width=True)
        
        # Display technical indicators if selected
        if show_technicals:
            st.markdown('<div class="sub-header">Technical Analysis</div>', unsafe_allow_html=True)
            
            # Calculate additional technical indicators
            data['RSI'] = calculate_rsi(data['Close'])
            data['MACD'], data['Signal'] = calculate_macd(data['Close'])
            
            # Plot technical indicators
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1]})
            
            # RSI plot
            ax1.plot(data.index[-100:], data['RSI'].iloc[-100:], 'purple', linewidth=1.5)
            ax1.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax1.axhline(30, color='green', linestyle='--', alpha=0.5)
            ax1.fill_between(data.index[-100:], data['RSI'].iloc[-100:], 70, 
                             where=(data['RSI'].iloc[-100:] >= 70), color='red', alpha=0.3)
            ax1.fill_between(data.index[-100:], data['RSI'].iloc[-100:], 30, 
                             where=(data['RSI'].iloc[-100:] <= 30), color='green', alpha=0.3)
            ax1.set_ylabel('RSI')
            ax1.set_title('Relative Strength Index (RSI)')
            ax1.grid(True, alpha=0.3)
            
            # MACD plot
            ax2.plot(data.index[-100:], data['MACD'].iloc[-100:], 'blue', linewidth=1.5, label='MACD')
            ax2.plot(data.index[-100:], data['Signal'].iloc[-100:], 'red', linewidth=1.5, label='Signal')
            ax2.bar(data.index[-100:], 
                    data['MACD'].iloc[-100:] - data['Signal'].iloc[-100:], 
                    color=['green' if x > 0 else 'red' for x in 
                           data['MACD'].iloc[-100:] - data['Signal'].iloc[-100:]], 
                    alpha=0.5, label='Histogram')
            ax2.set_ylabel('MACD')
            ax2.set_title('Moving Average Convergence Divergence (MACD)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add interpretation
            with st.expander("Technical Analysis Interpretation"):
                st.markdown("""
                ### RSI (Relative Strength Index)
                - **Above 70**: Stock may be overbought (selling pressure may be coming soon)
                - **Below 30**: Stock may be oversold (buying pressure may be coming soon)
                - **Current RSI**: {:.2f}
                
                ### MACD (Moving Average Convergence Divergence)
                - **MACD Line above Signal Line**: Bullish signal
                - **MACD Line below Signal Line**: Bearish signal
                - **MACD Histogram**: Shows the difference between MACD and Signal line
                """.format(data['RSI'].iloc[-1]))
        
        # Run prediction model if selected
        if show_predictions:
            st.markdown('<div class="sub-header">Price Predictions</div>', unsafe_allow_html=True)
            
            with st.spinner("Running hybrid prediction model... This may take a moment."):
                try:
                    test_data, hybrid_pred, metrics = train_models(data)
                    
                    # Show prediction chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(test_data.index, test_data['Close'], label='Actual Price', linewidth=2)
                    ax.plot(test_data.index, hybrid_pred, label='Hybrid Prediction', linewidth=2)
                    
                    # Add future predictions
                    future_dates = get_future_dates(test_data.index[-1], days=5)
                    future_prices = predict_future(None, test_data, days=5)
                    
                    # Convert to datetime for plotting
                    future_dates_dt = [pd.Timestamp(date) for date in future_dates]
                    
                    # Plot future predictions
                    ax.plot(future_dates_dt, future_prices, 'g--', label='5-Day Forecast', linewidth=2)
                    ax.fill_between(future_dates_dt, 
                                   [price * 0.98 for price in future_prices],
                                   [price * 1.02 for price in future_prices],
                                   color='green', alpha=0.2)
                    
                    # Format chart
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.set_title(f"{ticker} Price Predictions and 5-Day Forecast", fontsize=16)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Price ($)', fontsize=12)
                    ax.legend(fontsize=12)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Show future prediction table
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.markdown("### 5-Day Price Forecast")
                    
                    forecast_df = pd.DataFrame({
                        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                        'Predicted Price': [f"${price:.2f}" for price in future_prices],
                        'Change': [f"{((price/future_prices[i-1])-1)*100:.2f}%" if i > 0 else "N/A" 
                                  for i, price in enumerate(future_prices)]
                    })
                    
                    st.table(forecast_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error running prediction model: {e}")
                    st.error("Please try another stock or time period.")
    
    
        
    elif (page == "Market Overview"):
            st.markdown('<div class="sub-header">Market Overview</div>', unsafe_allow_html=True)
            
            # Select region for indices
            index_region = st.radio("Select Market Region:", ["US", "India", "Global"], horizontal=True)
            
            # Define indices based on selected region
            if index_region == "US":
                indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']
                indices_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
            elif index_region == "India":
                indices = ['^BSESN', '^NSEI', '^NSEBANK']
                indices_names = ['SENSEX', 'NIFTY 50', 'BANK NIFTY']
            else:  # Global
                indices = ['^NSEI', '^DJI', '^IXIC', '^N225', '^KS11', '^GDAXI']
                indices_names = ['SGX NIFTY', 'Dow Jones', 'NASDAQ', 'Nikkei 225', 'KOSPI', 'DAX']
            
            with st.spinner(f"Loading {index_region} market data..."):
                # Create layout
                col1, col2 = st.columns(2)
                
                for i, (index, name) in enumerate(zip(indices, indices_names)):
                    try:
                        index_data = yf.Ticker(index).history(period='1d')
                        if not index_data.empty:
                            current = index_data['Close'].iloc[-1]
                            prev = index_data['Close'].iloc[-2] if len(index_data) > 1 else current
                            change = current - prev
                            change_pct = (change / prev) * 100
                            
                            with col1 if i % 2 == 0 else col2:
                                st.metric(
                                    label=name,
                                    value=f"{current:.2f}",
                                    delta=f"{change:.2f} ({change_pct:.2f}%)"
                                )
                        else:
                            with col1 if i % 2 == 0 else col2:
                                st.metric(label=name, value="No data available")
                    except Exception as e:
                        with col1 if i % 2 == 0 else col2:
                            st.metric(label=name, value="Error loading data")
                            st.error(f"Error: {str(e)}")
                
                # ... rest of the market overview code ...
    if st.checkbox("Show Global Market Heat Map", value=True):
            st.markdown("### Global Market Heat Map")
            
            # Define global indices for heat map
            global_indices = {
                'US': ['^GSPC', '^DJI', '^IXIC'],
                'Europe': ['^GDAXI', '^FTSE', '^FCHI'],
                'Asia': ['^N225', '^HSI', '^BSESN', '^NSEI', '^KS11'],
            }
            
            index_performance = []
            
            with st.spinner("Generating global market heat map..."):
                for region, indices in global_indices.items():
                    for index in indices:
                        try:
                            # Get recent data (5 days)
                            index_data = yf.Ticker(index).history(period='5d')
                            if not index_data.empty:
                                start_price = index_data['Close'].iloc[0]
                                current_price = index_data['Close'].iloc[-1]
                                perf = ((current_price / start_price) - 1) * 100
                                
                                # Get proper index name
                                name_mapping = {
                                    '^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ',
                                    '^GDAXI': 'DAX', '^FTSE': 'FTSE 100', '^FCHI': 'CAC 40',
                                    '^N225': 'Nikkei 225', '^HSI': 'Hang Seng', 
                                    '^BSESN': 'SENSEX', '^NSEI': 'NIFTY 50', '^KS11': 'KOSPI'
                                }
                                
                                index_name = name_mapping.get(index, index)
                                
                                index_performance.append({
                                    'Region': region,
                                    'Index': index_name,
                                    'Performance': perf
                                })
                        except:
                            pass
        
    if index_performance:
            # Create DataFrame
            perf_df = pd.DataFrame(index_performance)
            
            # Create a pivot table for the heat map
            pivot_df = perf_df.pivot(index='Region', columns='Index', values='Performance')
            
            # Plot heat map
            fig, ax = plt.subplots(figsize=(14, 6))
            im = ax.imshow(pivot_df.values, cmap='RdYlGn')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(pivot_df.columns)))
            ax.set_yticks(np.arange(len(pivot_df.index)))
            ax.set_xticklabels(pivot_df.columns)
            ax.set_yticklabels(pivot_df.index)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            for i in range(len(pivot_df.index)):
                for j in range(len(pivot_df.columns)):
                    try:
                        value = pivot_df.iloc[i, j]
                        if not pd.isna(value):
                            text_color = 'black' if abs(value) < 3 else 'white'
                            ax.text(j, i, f"{value:.2f}%", ha="center", va="center", 
                                   color=text_color, fontweight='bold')
                    except:
                        pass
            
            plt.colorbar(im, label='5-Day Performance (%)')
            plt.title('Global Market Performance (5-Day)')
            plt.tight_layout()
            
            st.pyplot(fig)
    else:
            st.warning("Unable to load data for global market heat map")
            
            if sector_performance:
                # Create DataFrame and ensure 'YTD Return' exists and is valid
                sector_df = pd.DataFrame(sector_performance)

                if 'YTD Return' in sector_df.columns:
                    sector_df = sector_df.copy()  # Make sure it's not a view
                    sector_df = sector_df.sort_values(by='YTD Return', ascending=False)
                
                # Plot sector performance
                fig, ax = plt.subplots(figsize=(12, 6))
                bars = ax.bar(
                    sector_df['Sector'],
                    sector_df['YTD Return'],
                    color=['green' if ret > 0 else 'red' for ret in sector_df['YTD Return']]
                )
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height * (1.01 if height > 0 else 0.9),
                        f"{height:.1f}%",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=9
                    )
                
                plt.title('Sector Performance (YTD)')
                plt.ylabel('YTD Return (%)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                
                st.pyplot(fig)
            else:
                st.warning("Unable to load sector performance data")
    
    # Footer
    st.markdown('<div class="footer">Hybrid Stock Price Prediction App by Chaitanya Gaur | Data fetched by Yahoo Finance</div>', unsafe_allow_html=True)



def calculate_macd(prices, fast=12, slow=26, signal=9):
    # Calculate the Fast and Slow EMA
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # Calculate the MACD line
    macd = ema_fast - ema_slow
    
    # Calculate the Signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line

# Helper function for technical indicators
def calculate_rsi(prices, window=14):
    # Calculate price changes
    delta = prices.diff()
    
    # Separate positive and negative gains
    up = delta.copy()
    down = delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    down = down.abs()
    
    # Calculate the EWMA with min_periods to avoid comparison issues
    roll_up = up.ewm(span=window, min_periods=1).mean()
    roll_down = down.ewm(span=window, min_periods=1).mean()
    
    # Add a small number to avoid division by zero
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


if __name__ == "__main__":
    main()