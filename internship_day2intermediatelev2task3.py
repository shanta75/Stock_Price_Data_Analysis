import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Step 1: Load the cleaned dataset
file_path = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\cleaned_stock_prices.csv'
df = pd.read_csv(r'C:\Users\Shanta\PycharmProjects\PythonProject\intership1\cleaned_stock_prices.csv')

# Step 2: Select features for clustering
features = ['open', 'high', 'low', 'close', 'volume']
X = df[features]

# Step 3: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Elbow Method to find optimal number of clusters
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)

# Save the elbow plot
output_folder = 'C:\\Users\\Shanta\\PycharmProjects\\PythonProject\\intership1\\clustering_analysis'
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, 'elbow_method.png'))
plt.show()

# Step 5: Apply K-Means with optimal K (example: K=3 based on elbow plot)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to DataFrame
df['Cluster'] = clusters

# Step 6: Visualize Clusters (using 2 features: open vs close)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='open', y='close', hue='Cluster', data=df, palette='viridis')
plt.title(f'K-Means Clustering (K={optimal_k}): Open vs Close')
plt.xlabel('Open Price')
plt.ylabel('Close Price')
plt.legend(title='Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'clusters_open_close.png'))
plt.show()
