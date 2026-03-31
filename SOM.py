import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.cluster import KMeans
from collections import Counter

# Load standardized data
X_scaled_df = pd.read_csv("StdGasProperties.csv")
X = X_scaled_df.values

# -----------------------------
# SOM settings
# -----------------------------
som_x, som_y = 15, 15
num_iterations = 1000

som = MiniSom(
    x=som_x,
    y=som_y,
    input_len=X.shape[1],
    sigma=1.0,
    learning_rate=0.5,
    neighborhood_function='gaussian',
    random_seed=42
)

som.random_weights_init(X)
som.train_random(X, num_iterations)

print("SOM settings:")
print(f"Grid size: {som_x}x{som_y}")
print("Neighborhood function: gaussian")
print("Initial learning rate: 0.5 (decays during training)")
print(f"Number of iterations: {num_iterations}")
print("Similarity metric: Euclidean distance")

# -----------------------------
# Extract neuron prototypes
# -----------------------------
prototypes = som.get_weights().reshape(som_x * som_y, X.shape[1])

print("\nExtracted neuron prototypes shape:")
print(prototypes.shape)

# -----------------------------
# K-means on prototypes
# -----------------------------
kmeans_proto = KMeans(
    n_clusters=3,
    init='k-means++',
    random_state=42,
    n_init=10
)

prototype_labels = kmeans_proto.fit_predict(prototypes)

print("\nK-means cluster centroids on SOM prototypes:")
print(kmeans_proto.cluster_centers_)

# -----------------------------
# Assign each sample the cluster of its BMU
# -----------------------------
data_cluster_labels = []

for x in X:
    bmu = som.winner(x)
    bmu_index = bmu[0] * som_y + bmu[1]
    cluster_label = prototype_labels[bmu_index]
    data_cluster_labels.append(cluster_label)

data_cluster_labels = np.array(data_cluster_labels)

# -----------------------------
# Cluster sizes
# -----------------------------
cluster_sizes = Counter(data_cluster_labels)

print("\nFinal cluster sizes:")
for cluster_id in sorted(cluster_sizes.keys()):
    print(f"Cluster {cluster_id}: {cluster_sizes[cluster_id]} samples")

# -----------------------------
# Final cluster centroids from data
# -----------------------------
final_centroids = []

for k in range(3):
    cluster_points = X[data_cluster_labels == k]
    centroid = cluster_points.mean(axis=0)
    final_centroids.append(centroid)

final_centroids = np.array(final_centroids)

print("\nFinal cluster centroids (data-based):")
print(final_centroids)

# Save SOM labels for wobbe-index.py
np.savetxt("som_labels.csv", data_cluster_labels, fmt="%d")

# -------------------------------
#  Calculate silhouette score
# -------------------------------

from sklearn.metrics import silhouette_score

som_silhouette = silhouette_score(X, data_cluster_labels)
print("\nSilhouette Score (SOM-derived clusters):", som_silhouette)

# -----------------------------
# Plot clusters using PCA
# -----------------------------
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# print("\nPlotting clusters using PCA projection...")

# # Reduce dimensions to 2D for visualization
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # Transform centroids to PCA space
# centroids_pca = pca.transform(final_centroids)

# plt.figure(figsize=(8,6))

# scatter = plt.scatter(
#     X_pca[:,0],
#     X_pca[:,1],
#     c=data_cluster_labels,
#     cmap='viridis',
#     s=2,
#     alpha=0.5
# )

# # Plot centroids
# plt.scatter(
#     centroids_pca[:,0],
#     centroids_pca[:,1],
#     c='red',
#     s=200,
#     marker='X',
#     label='Cluster Centroids'
# )

# plt.title("SOM Clustering (PCA Projection)")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")

# plt.legend()
# plt.colorbar(scatter, label="Cluster ID")

# plt.show()

 