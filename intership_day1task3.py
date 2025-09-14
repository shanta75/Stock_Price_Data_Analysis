import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df = pd.read_csv(r'C:\Users\Shanta\PycharmProjects\PythonProject\intership1\cleaned_stock_prices.csv')

# Ensure 'date' column is datetime type
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Create output folder for images
import os
output_folder = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\eda_plots'
os.makedirs(output_folder, exist_ok=True)

# --- Bar Plot: Number of records per symbol ---
plt.figure(figsize=(12, 6))
symbol_counts = df['symbol'].value_counts().head(10)  # Top 10 symbols
sns.barplot(x=symbol_counts.index, y=symbol_counts.values, palette='viridis')
plt.title('Top 10 Symbols by Record Count')
plt.xlabel('Symbol')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/barplot_symbol_count.png')
plt.show()

# --- Line Chart: Average Close Price over Time (aggregated by date) ---
avg_close = df.groupby('date')['close'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(x='date', y='close', data=avg_close, marker='o')
plt.title('Average Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Average Close Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_folder}/linechart_avg_close.png')
plt.show()

# --- Scatter Plot: Open vs Close Prices ---
plt.figure(figsize=(10, 6))
sns.scatterplot(x='open', y='close', data=df, alpha=0.5)
plt.title('Scatter Plot: Open vs Close Prices')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.tight_layout()
plt.savefig(f'{output_folder}/scatter_open_vs_close.png')
plt.show()
