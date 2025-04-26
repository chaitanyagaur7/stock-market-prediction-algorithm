# Stock Market Analysis & Prediction App

This is a **Streamlit** application built for **Stock Market Analysis and Price Prediction**.  
It uses real-time stock data, visualizes key technical indicators like **RSI** and **MACD**, and predicts future stock prices through a **hybrid machine learning model** combining **LSTM**, **SVM**, and **Linear Regression**.

---

## Live Application

[Click here to access the live app](https://stock-market-prediction-algorithm-chaitanya.streamlit.app)

---

## Features

- **Real-Time Stock Data Retrieval**  
  Fetches the latest stock data dynamically using `yfinance`, with progress tracking.

- **Interactive Visualizations**  
  Visualizes key technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)

- **Sector-Specific Stock Exploration**  
  Explore popular stocks across the following sectors:
  - Technology
  - Finance
  - Healthcare
  - Energy
  - Consumer Goods

- **Future Stock Price Prediction**  
  Predicts future stock movements using a hybrid model:
  - LSTM (Long Short-Term Memory)
  - Support Vector Machine (SVM)
  - Linear Regression

- **Seamless User Experience**  
  Built with Streamlit to provide a fast, interactive, and intuitive user interface.

---

## Project Focus Areas

- Time Series Forecasting for stock price prediction
- Financial Data Visualization with technical indicators
- Development and deployment of a hybrid Machine Learning model
- Building and deploying Streamlit applications

---

## Tech Stack

- Python
- Streamlit
- yfinance
- TensorFlow / Keras
- scikit-learn
- Matplotlib / Plotly

---

## Installation

To run the application locally:

```bash
# Clone the repository
git clone https://github.com/your-username/stock-market-prediction-app.git

# Navigate into the project directory
cd stock-market-prediction-app

# Install required dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

