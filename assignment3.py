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