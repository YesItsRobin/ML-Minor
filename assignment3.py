import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans

df = pd.read_csv('card_transdata.csv', header = 0)

# Preprocessing scaled data
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']])
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:3])

# K-means clustering on scaled data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
'''
# Initialize an empty list to store the within-cluster sum of squares (inertia)
inertia = []

# Define the range of clusters you want to try
num_clusters = range(1, 21)  # You can adjust this range as needed

# Fit K-means clustering for each value of k
for k in num_clusters:
    print(k)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(num_clusters, inertia, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

#Best was 10 clusters

'''
# K-means clustering on scaled data
kmeans = KMeans(n_clusters=10, random_state=42)
df_scaled['cluster'] = kmeans.fit_predict(df_scaled)

df_filtered = df_scaled[df_scaled['cluster'] != 5]

# Count the number of fraud datapoints in each cluster
fraud_counts = df['fraud'][df_filtered.index].groupby(df_filtered['cluster']).sum()

cluster_profiles = df_filtered.groupby(by='cluster')[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']].mean()
cluster_counts = df_filtered['cluster'].value_counts().sort_index()

cluster_profiles.plot(kind='bar', figsize=(10, 6))

print(cluster_counts)
#print(fraud_counts)
plt.title('Cluster Profiles')
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

# Expectation Maximization Algorithm
from sklearn.mixture import GaussianMixture

# Initialize an empty list to store the log-likelihood scores
log_likelihoods = []
'''
# Define the range of clusters you want to try
num_clusters = range(1, 11)  # You can adjust this range as needed

# Fit Gaussian Mixture Model clustering for each value of k
for k in num_clusters:
    print(k)
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(df_scaled)
    log_likelihoods.append(gmm.score(df_scaled))

# Plot the log-likelihoods scores
plt.figure(figsize=(8, 6))
plt.plot(num_clusters, log_likelihoods, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Log-likelihood')
plt.title('GMM Log-likelihood vs. k')
plt.grid(True)
plt.show()

#Best was 6 clusters
gmm = GaussianMixture(n_components=6, random_state=42)
gmm.fit(df_scaled)
# show cluster profiles and size
df_scaled['cluster'] = gmm.predict(df_scaled)
cluster_profiles = df_scaled.groupby(by='cluster')[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']].mean()
cluster_profiles.plot(kind='bar', figsize=(10, 6))
cluster_count = df_scaled['cluster'].value_counts()
print(cluster_count)
plt.title('Cluster Profiles')
plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.xticks(rotation=0)
plt.show()

'''
