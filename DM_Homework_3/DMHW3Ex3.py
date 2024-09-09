
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# Load your dataset (replace 'your_dataset.csv' with your actual dataset file)
data = pd.read_csv('amazon_productsClean.tsv', sep='\t')
dataFE = pd.read_csv('amazon_productsClean.tsv', sep='\t')

# Display the first few rows of the dataset to understand its structure


# Select relevant features for clustering (adjust as needed)
selected_features = data[['Price', 'Stars', 'Number of Reviews']]
selected_features.fillna(selected_features.mean(), inplace=True)
selected_features.dropna(inplace=True)

# Determine the optimal number of clusters using the elbow method
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(selected_features)
    wcss.append(kmeans.inertia_)


# Plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# Choose the optimal number of clusters based on the elbow method
optimal_clusters = 3 
optimal_clustersFE = 3

# Apply KMeans clustering algorithm
start_time_original_kmeans = time.time()

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
data['cluster'] = kmeans.fit_predict(selected_features)

end_time_original_kmeans = time.time()
elapsed_time_original_kmeans = end_time_original_kmeans - start_time_original_kmeans




# Feature Engineering Section
start_time_originalFE = time.time()

selected_features_log = selected_features.copy()
selected_features_log[['Price', 'Number of Reviews']] = selected_features_log[['Price', 'Number of Reviews']].applymap(lambda x: max(0, x))  # Ensure non-negative values
selected_features_log[['Price', 'Number of Reviews']] = selected_features_log[['Price', 'Number of Reviews']].applymap(lambda x: 0 if x == 0 else np.log(x) + 1)

scaler_standard = StandardScaler()
scaled_data = scaler_standard.fit_transform(selected_features_log)

scaler_minmax = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler_minmax.fit_transform(scaled_data)

# Map the normalized values to the desired range
normalized_data[:, 0] *= 15  # Normalize 'Price' into 15
normalized_data[:, 1] *= 5   # Normalize 'Stars' into 5
normalized_data[:, 2] *= 9   # Normalize 'Number of Reviews' into 5

# Update the dataframe with the normalized values
dataFE[['Price', 'Stars', 'Number of Reviews']] = normalized_data

end_time_originalFE = time.time()
elapsed_time_original = end_time_originalFE - start_time_originalFE


wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(normalized_data)
    wcss.append(kmeans.inertia_)



# Plot the elbow curve
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method FE')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

start_time_fe = time.time()

kmeans = KMeans(n_clusters=optimal_clustersFE, init='k-means++', random_state=42)
dataFE['cluster'] = kmeans.fit_predict(normalized_data)

end_time_fe = time.time()
elapsed_time_fe = end_time_fe - start_time_fe

#comment this lines if u want to see the modified dataset
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
dataFE['cluster'] = kmeans.fit_predict(normalized_data)
dataFE[['Price', 'Stars', 'Number of Reviews']] = data[['Price', 'Stars', 'Number of Reviews']].copy()


output_file = 'clustered_data.csv'
output_fileFE = 'clustered_dataFE.csv'

data[['Price', 'Stars', 'Number of Reviews', 'cluster']].to_csv(output_file, index=False)
dataFE[['Price', 'Stars', 'Number of Reviews', 'cluster']].to_csv(output_fileFE, index=False)

print(f"Clustered data saved to {output_file}")
print(f"Clustered data saved to {output_fileFE}")

# Display the cluster assignments
print("Cluster Assignments for Original Data:")
print(data[['Price', 'Stars', 'Number of Reviews', 'cluster']])
print("\nCluster Assignments for Feature-Engineered Data:")
print(dataFE[['Price', 'Stars', 'Number of Reviews', 'cluster']])

# Cluster Counts
cluster_counts_original = data['cluster'].value_counts().sort_index()
print("\nCluster Counts for Original Data:")
print(cluster_counts_original)

# For the feature-engineered dataset
cluster_counts_fe = dataFE['cluster'].value_counts().sort_index()
print("\nCluster Counts for Feature-Engineered Data:")
print(cluster_counts_fe)

# 3D Scatter Plots
fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter3D(selected_features.values[:, 0], selected_features.values[:, 1], selected_features.values[:, 2], c=data['cluster'], cmap='viridis')
ax1.set_title('Original Data Clusters')
ax1.set_xlabel('Price')
ax1.set_ylabel('Stars')
ax1.set_zlabel('Number of Reviews')

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(dataFE['Price'], dataFE['Stars'], dataFE['Number of Reviews'], c=dataFE['cluster'], cmap='viridis')
ax2.set_title('Normalized Data Clusters')
ax2.set_xlabel('Price')
ax2.set_ylabel('Stars')
ax2.set_zlabel('Number of Reviews')

plt.show()

print(f"Time taken for KMeans without feature engineering: {elapsed_time_original_kmeans:.2f} seconds")
print(f"Time taken for feature engineering: {elapsed_time_original:.2f} seconds")
print(f"Time taken for feature engineering K-means: {elapsed_time_fe:.2f} seconds")