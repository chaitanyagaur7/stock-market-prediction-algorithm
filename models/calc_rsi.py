# models/calc_rsi.py

import pandas as pd

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
