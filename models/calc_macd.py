def calculate_macd(prices, fast=12, slow=26, signal=9):
    # Calculate the Fast and Slow EMA
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    
    # Calculate the MACD line
    macd = ema_fast - ema_slow
    
    # Calculate the Signal line
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    
    return macd, signal_line