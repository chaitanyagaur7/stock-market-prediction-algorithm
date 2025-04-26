def train_models(data):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, r2_score
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    import streamlit as st
    from sklearn.linear_model import LinearRegression
    
    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
    
    # Find Close column
    close_candidates = [col for col in data.columns if 'close' in col.lower()]
    if not close_candidates:
        raise ValueError("No column with 'Close' found in the dataset.")
    
    close_col = close_candidates[0]
    st.write(f"Using close column: {close_col}")
    
    # Standardize column names
    data['Close'] = data[close_col]
    
    # Feature engineering - create more informative features
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['Returns'] = data['Close'].pct_change()
    data['Returns_MA5'] = data['Returns'].rolling(window=5).mean()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    
    # Calculate price momentum
    data['Momentum'] = data['Close'] - data['Close'].shift(5)
    
    # Clean data
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    
    if len(data) < 150:
        raise ValueError("Not enough data after cleaning to split into train/test.")
    
    # Split data: use more training data
    train_data, test_data = data.iloc[:-100].copy(), data.iloc[-100:].copy()
    
    # Save original Close prices
    original_train_close = train_data['Close'].values
    original_test_close = test_data['Close'].values
    
    # Define features for models
    basic_features = ['Returns', 'MA50', 'MA20', 'MA10', 'Returns_MA5', 'Volatility', 'Momentum']
    
    # Use MinMaxScaler instead of StandardScaler for price data
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler.fit(train_data[['Close']])
    
    # Feature scaler for other features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler.fit(train_data[basic_features])
    
    # Scale data
    train_data[['Close']] = price_scaler.transform(train_data[['Close']])
    test_data[['Close']] = price_scaler.transform(test_data[['Close']])
    
    train_data[basic_features] = feature_scaler.transform(train_data[basic_features])
    test_data[basic_features] = feature_scaler.transform(test_data[basic_features])
    
    # Create sequences for LSTM
    def create_sequences(data, features, target, seq_length=10):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[features].values[i:i+seq_length])
            y.append(data[target].values[i+seq_length])
        return np.array(X), np.array(y)
    
    # Prepare LSTM data with a longer sequence
    seq_length = 20
    lstm_features = basic_features
    X_train, y_train = create_sequences(train_data, lstm_features, 'Close', seq_length)
    
    # Train SVR with more features
    svr = SVR(kernel='rbf', C=10, gamma='scale', epsilon=0.1)
    svr.fit(train_data[basic_features], train_data['Close'])
    
    # Build more powerful LSTM
    lstm = Sequential()
    lstm.add(LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm.add(Dropout(0.2))
    lstm.add(LSTM(64))
    lstm.add(Dropout(0.2))
    lstm.add(Dense(32))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    
    # Train with more epochs and early stopping
    from keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(monitor='loss', patience=5)
    lstm.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
    
    # Train Linear Regression with more features
    reg = LinearRegression()
    reg.fit(train_data[basic_features], train_data['Close'])
    
    # Make predictions
    svr_pred = svr.predict(test_data[basic_features])
    
    # For LSTM, create proper test sequences
    X_test_sequences = []
    for i in range(len(test_data) - seq_length):
        X_test_sequences.append(test_data[lstm_features].values[i:i+seq_length])
    
    if X_test_sequences:
        X_test = np.array(X_test_sequences)
        lstm_pred_partial = lstm.predict(X_test).flatten()
        
        # Pad beginning with values from training prediction
        padding = np.full(seq_length, lstm_pred_partial[0])
        lstm_pred = np.concatenate((padding, lstm_pred_partial))
    else:
        lstm_pred = np.full(len(test_data), train_data['Close'].mean())
    
    reg_pred = reg.predict(test_data[basic_features])
    
    # Combine models with weighted average (give LSTM more weight)
    hybrid_pred = (0.3 * svr_pred + 0.5 * lstm_pred + 0.2 * reg_pred)
    
    # Inverse transform to get back to original scale
    svr_pred_original = price_scaler.inverse_transform(svr_pred.reshape(-1, 1)).flatten()
    lstm_pred_original = price_scaler.inverse_transform(lstm_pred.reshape(-1, 1)).flatten()
    reg_pred_original = price_scaler.inverse_transform(reg_pred.reshape(-1, 1)).flatten()
    hybrid_pred_original = price_scaler.inverse_transform(hybrid_pred.reshape(-1, 1)).flatten()
    
    # Calculate metrics on original scale
    metrics = {
        "svr": {
            "mse": mean_squared_error(original_test_close, svr_pred_original), 
            "r2": r2_score(original_test_close, svr_pred_original)
        },
        "lstm": {
            "mse": mean_squared_error(original_test_close, lstm_pred_original), 
            "r2": r2_score(original_test_close, lstm_pred_original)
        },
        "regression": {
            "mse": mean_squared_error(original_test_close, reg_pred_original), 
            "r2": r2_score(original_test_close, reg_pred_original)
        },
        "hybrid": {
            "mse": mean_squared_error(original_test_close, hybrid_pred_original), 
            "r2": r2_score(original_test_close, hybrid_pred_original)
        },
    }
    
    # Restore original Close values for visualization
    test_data['Close'] = original_test_close
    
    # Also return individual model predictions for comparison
    return test_data, hybrid_pred_original, metrics