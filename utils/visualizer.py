import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

def plot_stock_data(data, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    data['Close'].plot(label='Close Price', ax=ax)
    data['MA50'].plot(label='50-day MA', ax=ax)
    ax.set_title(f"{ticker} Stock Price and MA50")
    ax.legend()
    return fig

import plotly.graph_objects as go

def plot_predictions(dates, actual_prices, predicted_prices):
    fig = go.Figure()

    # Actual Prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_prices,
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Predicted Prices
    fig.add_trace(go.Scatter(
        x=dates,
        y=predicted_prices,
        mode='lines+markers',  # adding markers helps hover visibility
        name='Predicted Price',
        line=dict(color='red')
    ))

    # Layout Settings
    fig.update_layout(
        title='Actual vs Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Stock Price',
        hovermode='x unified',  # <-- This shows the price info when hovering
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)
