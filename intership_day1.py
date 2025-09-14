import pandas as pd

# Step 1: Load the dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\2) Stock Prices Data Set.csv'
df = pd.read_csv(r'C:\Users\Shanta\Downloads\2) Stock Prices Data Set.csv')

# Step 2: Identify missing values
print("Initial missing values in each column:")
print(df.isnull().sum())

# Step 3: Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].mean())

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Step 4: Remove duplicate rows
initial_count = len(df)
df = df.drop_duplicates()
final_count = len(df)
print(f"\nDuplicates removed: {initial_count - final_count}")

# Step 5.1: Standardize Date Column
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])

# Step 5.2: Standardize categorical variables
categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    df[col] = df[col].str.strip().str.lower()

# Step 6: Save the cleaned dataset
cleaned_file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df.to_csv(cleaned_file_path, index=False)

print(f"\nCleaned dataset saved at: {cleaned_file_path}")
