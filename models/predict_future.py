import numpy as np
from datetime import timedelta, datetime
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