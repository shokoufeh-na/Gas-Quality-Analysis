import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load standardized data
df = pd.read_csv("StdGasProperties.csv")

# Fit K-means
kmeans = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    random_state=0
)
kmeans.fit(df)

# 1. Initialization method
print("Initialization method:", kmeans.init)

# 2. Initial centroids (approximation)
# We approximate them by running one iteration manually
kmeans_temp = KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=1,
    max_iter=1,
    random_state=0
).fit(df)

print("\nInitial centroids (from first iteration):")
print(kmeans_temp.cluster_centers_)

# 3. Number of iterations
print("\nIterations until convergence:", kmeans.n_iter_)

# 4. Final centroids
print("\nFinal centroids:")
print(kmeans.cluster_centers_)

# 5. Cluster variances (WCSS)
labels = kmeans.labels_
wcss = []
for k in range(3):
    cluster_points = df[labels == k]
    centroid = kmeans.cluster_centers_[k]
    wcss_k = ((cluster_points - centroid)**2).sum().sum()
    wcss.append(wcss_k)

print("\nWithin-cluster sum of squares:", wcss)

# 6. Number of samples per cluster
cluster_sizes = pd.Series(labels).value_counts().sort_index()
print("\nSamples per cluster:")
print(cluster_sizes)


# Save K-Means labels for wobbe-index.py
np.savetxt("kmeans_labels.csv", labels, fmt="%d")


# -------------------------------
#  Calculate silhouette score
# -------------------------------


from sklearn.metrics import silhouette_score

# Silhouette score for K-Means
sil_score = silhouette_score(df, labels)

print("\nSilhouette Score (K-Means):", sil_score)


# -------------------------------
#  ploting
# -------------------------------

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA

# # Load standardized data
# df = pd.read_csv("StdGasProperties.csv")

# # Fit K-means
# kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
# labels = kmeans.fit_predict(df)

# # PCA to 2D
# pca = PCA(n_components=2)
# pca_data = pca.fit_transform(df)

# plt.figure(figsize=(8,6))
# plt.scatter(pca_data[:,0], pca_data[:,1], c=labels, cmap='viridis', s=5)
# plt.title("K-Means Clusters (PCA 2D Projection)")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.colorbar(label="Cluster")
# plt.show()


# import seaborn as sns

# df_plot = df.copy()
# df_plot["Cluster"] = labels

# sns.pairplot(df_plot, hue="Cluster", diag_kind="kde",
#              plot_kws={"s": 10, "alpha": 0.6})
# plt.suptitle("K-Means Clusters Across Chemical Features", y=1.02)
# plt.show()
