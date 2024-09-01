import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, DBSCAN

moon_cluster, _ = make_moons(n_samples=500, noise=0.05)
circular_cluster, _ = make_blobs(n_samples=100, centers=[(1.5, 1)], cluster_std=0.1)

data1 = np.vstack([moon_cluster, circular_cluster, np.random.uniform(low=-1, high=2.5, size=(20, 2))])


dense_circular_cluster, _ = make_blobs(n_samples=100, centers=[(0, 0)], cluster_std=0.1)
sparse_circular_cluster, _ = make_blobs(n_samples=100, centers=[(3, 3)], cluster_std=0.6)

data2 = np.vstack([dense_circular_cluster, sparse_circular_cluster, np.random.uniform(low=-3, high=6, size=(20, 2))])

# Dataset 1

# Kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels_kmeans = kmeans.fit_predict(data1)

# DBSCAN
# TODO play with the eps and min_samples values to see if the clustering gets better or worse
dbscan = DBSCAN(eps=0.2, min_samples=5)
labels_dbscan = dbscan.fit_predict(data1)

figure1, axes = plt.subplots(1, 3, figsize=(18,6))

axes[0].scatter(data1[:, 0], data1[:, 1], s=10)
axes[0].set_title("Original Dataset")

axes[1].scatter(data1[:, 0], data1[:, 1], c=labels_kmeans, cmap='viridis', s=10)
axes[1].set_title("K-Means Clustering Result")

axes[2].scatter(data1[:, 0], data1[:, 1], c=labels_dbscan, cmap='viridis', s=10)
axes[2].set_title("DBSCAN Clustering Result")

# Dataset 2

kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(data2)

# ifi eps = 0.2 it doesn't work properly 
dbscan = DBSCAN(eps=0.5,  min_samples=5)
labels_dbscan = dbscan.fit_predict(data2)

figure2, axes = plt.subplots(1, 3, figsize=(18,6))

axes[0].scatter(data2[:, 0], data2[:, 1], s=10)
axes[0].set_title("Original Dataset")

axes[1].scatter(data2[:, 0], data2[:, 1], c=labels_kmeans, cmap='viridis', s=10)
axes[1].set_title("K-Means Clustering Result")

axes[2].scatter(data2[:, 0], data2[:, 1], c=labels_dbscan, cmap='viridis', s=10)
axes[2].set_title("DBSCAN Clustering Result")

plt.show()
