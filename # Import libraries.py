# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import yfinance as yf

# Download stock data
data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

# Use closing price
data = data[['Close']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create dataset (X = previous 5 days, y = next day)
X = []
y = []

for i in range(5, len(scaled_data)):
    X.append(scaled_data[i-5:i, 0])
    y.append(scaled_data[i, 0])

X, y = np.array(X), np.array(y)

# Split data (80% training, 20% testing)
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predicted = model.predict(X_test)

# Convert back to original values
predicted = scaler.inverse_transform(predicted.reshape(-1,1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# Plot results
plt.plot(y_test_actual, label='Actual Price')
plt.plot(predicted, label='Predicted Price')
plt.legend()
plt.title("Stock Price Prediction (Without TensorFlow)")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()