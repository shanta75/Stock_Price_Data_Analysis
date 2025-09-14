import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the cleaned dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df = pd.read_csv(r'C:\Users\Shanta\PycharmProjects\PythonProject\intership1\cleaned_stock_prices.csv')

# Step 2: Define features and target
X = df[['open']]    # Independent variable
y = df['close']     # Target variable

# Step 3: Split into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Fit Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Model Coefficients
print(f"Intercept (b0): {model.intercept_}")
print(f"Coefficient (b1): {model.coef_[0]}")

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R²) Score: {r2:.4f}")

# Step 8: Plot Regression Line
plt.figure(figsize=(10, 6))
sns.scatterplot(x='open', y='close', data=df, alpha=0.5, label='Actual Data')
sns.lineplot(x=X_test['open'], y=y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Open Price vs Close Price')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()

# Save plot as image
plt.savefig('C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\regression_open_close.png')
plt.show()
