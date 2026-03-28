# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore
import yfinance as yf

# Download stock data (example: Apple)
data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

# Use closing price
data = data[['Close']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create training dataset
X_train = []
y_train = []

for i in range(60, len(scaled_data)):
    X_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32)

# Prepare test data
test_data = scaled_data[-60:]
X_test = []

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Predict
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)

# Plot results
plt.plot(data.index[-len(predicted_price):], predicted_price, label='Predicted Price')
plt.plot(data.index[-len(predicted_price):], data['Close'][-len(predicted_price):], label='Actual Price')
plt.legend()
plt.title("Stock Price Prediction")
plt.show()
