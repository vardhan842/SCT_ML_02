import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Sample synthetic data (replace with actual purchase history)
data = {
    'CustomerID': [1, 2, 3, 4, 5],
    'Annual_Spend': [1500, 2400, 3200, 1800, 2900],
    'Purchase_Frequency': [10, 18, 25, 12, 20]
}
df = pd.DataFrame(data)

# Features for clustering
X = df[['Annual_Spend', 'Purchase_Frequency']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters
plt.figure(figsize=(8, 5))
plt.scatter(df['Annual_Spend'], df['Purchase_Frequency'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Annual Spend')
plt.ylabel('Purchase Frequency')
plt.title('Customer Segments using K-Means Clustering')
plt.grid(True)
plt.show()