# stock_price_predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load your dataset (Example: Apple stock prices)
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/AAPL.csv')

# Use 'Date' and 'Close' columns
df['Date'] = pd.to_datetime(df['Date'])
df['Date_ordinal'] = df['Date'].map(lambda date: date.toordinal())

X = df[['Date_ordinal']]
y = df['Close']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], label='Actual Prices')
plt.plot(df['Date'].iloc[-len(y_pred):], y_pred, label='Predicted Prices', color='red')
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()
