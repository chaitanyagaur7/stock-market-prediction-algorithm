import yfinance as yf
import pandas as pd

def load_stock_data(ticker, period='5y'):
    data = yf.download(ticker, period=period)
    data['Returns'] = data['Close'].pct_change()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    return data.dropna()
