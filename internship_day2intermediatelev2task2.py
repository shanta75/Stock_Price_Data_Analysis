import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

# Step 1: Load cleaned dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df = pd.read_csv(r'C:\Users\Shanta\PycharmProjects\PythonProject\intership1\cleaned_stock_prices.csv')

# Ensure 'date' column is datetime type and set as index
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.set_index('date')

# Sort by date just in case
df = df.sort_index()

# Step 2: Plot Time-Series Data (Close Price over Time)
plt.figure(figsize=(14, 6))
df['close'].plot(title='Stock Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.tight_layout()

# Save Time-Series Plot
output_folder = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\time_series_analysis'
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, 'time_series_close_price.png'))
plt.show()

# Step 3: Decompose Time Series into Trend, Seasonality, and Residuals
# Note: freq=30 assumes monthly seasonality if data is daily
decomposition = seasonal_decompose(df['close'], model='additive', period=30)

plt.figure(figsize=(14, 10))
decomposition.plot()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'time_series_decomposition.png'))
plt.show()

# Step 4: Moving Average Smoothing (Window of 30 days)
df['close_ma30'] = df['close'].rolling(window=30).mean()

plt.figure(figsize=(14, 6))
plt.plot(df['close'], label='Original Close Price')
plt.plot(df['close_ma30'], label='30-Day Moving Average', color='orange')
plt.title('Moving Average Smoothing (30 Days)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'moving_average_smoothing.png'))
plt.show()
