import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the cleaned dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df = pd.read_csv(r'C:\Users\Shanta\PycharmProjects\PythonProject\intership1\cleaned_stock_prices.csv')

# Set pandas option to display all rows
pd.set_option('display.max_rows', None)

# Preview the entire dataset
print("First 5 rows of the dataset:")
print(df.head())

# Step 2: Summary Statistics
mean_values = df.mean(numeric_only=True)
median_values = df.median(numeric_only=True)
mode_values = df.mode().iloc[0]  # Take first row of mode
std_dev_values = df.std(numeric_only=True)

print("\nMean values:\n", mean_values)
print("\nMedian values:\n", median_values)
print("\nMode values:\n", mode_values)
print("\nStandard Deviation values:\n", std_dev_values)

# Step 3.1: Histogram of Numeric Columns
df.hist(figsize=(12, 8), bins=30)
plt.suptitle('Histograms of Numerical Features')
plt.show()

# Step 3.2: Boxplot of Numeric Columns
plt.figure(figsize=(12, 8))
sns.boxplot(data=df.select_dtypes(include='number'))
plt.title('Boxplots of Numerical Features')
plt.show()

# Step 3.3: Scatter Plot (Open vs Close)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='open', y='close', data=df)
plt.title('Scatter Plot: Open vs Close Prices')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.show()

# Step 4: Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
