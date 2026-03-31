import pandas as pd
import numpy as np

df = pd.read_csv("GasProperties.csv")

low_thr = df["Idx"].quantile(0.33)
high_thr = df["Idx"].quantile(0.66)

print("Wobbe thresholds:")
print("Regular <", low_thr)
print("Medium <", high_thr)

def wobbe_class(x):
    if x <= low_thr:
        return "Regular"
    elif x <= high_thr:
        return "Medium"
    else:
        return "Premium"

df["Gas_Quality"] = df["Idx"].apply(wobbe_class)

print(df["Gas_Quality"].value_counts())

stats = df.groupby("Gas_Quality")[["T","P","TC","SV"]].agg(["mean","var"])

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print(stats)

# ---------------------------------------------------------
# 1. Original dataset means and stds (before standardization)
# ---------------------------------------------------------
mu = np.array([303.150000, 121.100000, 0.034696, 415.243807])
sigma = np.array([34.156543, 95.286764, 0.004071, 22.532554])

# ---------------------------------------------------------
# 2. Class means (original units)
# ---------------------------------------------------------
class_means = {
    "Medium":  np.array([303.199278, 121.506210, 0.034660, 415.128573]),
    "Premium": np.array([303.167857, 120.815287, 0.034685, 412.046145]),
    "Regular": np.array([303.082323, 120.987126, 0.034743, 418.653580])
}

# ---------------------------------------------------------
# 3. Standardize class means
# ---------------------------------------------------------
class_means_std = {cls: (vals - mu) / sigma for cls, vals in class_means.items()}

print("Standardized class means:\n")
for cls, z in class_means_std.items():
    print(f"{cls}: {z}\n")

# ---------------------------------------------------------
# 4. Cluster centers (already standardized)
# ---------------------------------------------------------

# K-Means
K_centroids = np.array([
    [-0.95549736, -0.50820351, -1.09056186, -0.74687821],
    [-0.28086133,  1.32265129,  0.19999315, -0.66232277],
    [ 0.95231794, -0.26228490,  0.81506303,  0.97564496]
])

# EM Means
EM_means = np.array([
    [ 1.02910681, -0.18180898,  0.91829939,  1.02764859],
    [-0.61887752,  0.74748725, -0.27402545, -0.80043724],
    [-0.61475394, -0.94099044, -1.00615450, -0.31382122]
])

# SOM centroids
SOM_centroids = np.array([
    [ 0.95123331, -0.14129656,  0.84095288,  0.93576860],
    [-1.24890276, -0.65463707, -1.43432690, -1.00954403],
    [-0.49869884,  0.52719914, -0.26278625, -0.60873511]
])

# ---------------------------------------------------------
# 5. Function to compute Euclidean distances
# ---------------------------------------------------------
def compute_distances(class_means_std, centers):
    rows = []
    for cls, z in class_means_std.items():
        dists = np.linalg.norm(centers - z, axis=1)
        rows.append([cls] + list(dists))
    return pd.DataFrame(rows, columns=["Class"] + [f"Cluster_{i}" for i in range(centers.shape[0])])

# ---------------------------------------------------------
# 6. Compute distance tables
# ---------------------------------------------------------
print("\n=== K-Means Distances ===\n")
print(compute_distances(class_means_std, K_centroids))

print("\n=== EM/GMM Distances ===\n")
print(compute_distances(class_means_std, EM_means))

print("\n=== SOM Distances ===\n")
print(compute_distances(class_means_std, SOM_centroids))

#  ---------------------------------------------------------
# 7. NOW ADD ARI CODE HERE
# ---------------------------------------------------------
from sklearn.metrics import adjusted_rand_score

# True labels from Wobbe Index classes
true_labels = df["Gas_Quality"].map({"Regular":0, "Medium":1, "Premium":2}).values
# -----------------------
#  Load clustering labels
# -----------------------
kmeans_labels = np.loadtxt("kmeans_labels.csv", dtype=int)
em_labels     = np.loadtxt("em_labels.csv", dtype=int)
som_labels    = np.loadtxt("som_labels.csv", dtype=int)
 
# ---------------------------------------------------------
# After computing kmeans_labels, em_labels, som_labels
# ---------------------------------------------------------

from sklearn.metrics import adjusted_rand_score

true_labels = df["Gas_Quality"].map({"Regular":0, "Medium":1, "Premium":2}).values

ari_kmeans = adjusted_rand_score(true_labels, kmeans_labels)
ari_em     = adjusted_rand_score(true_labels, em_labels)
ari_som    = adjusted_rand_score(true_labels, som_labels)

print("\n=== Adjusted Rand Index (ARI) ===")
print("K-Means ARI:", ari_kmeans)
print("EM/GMM ARI:", ari_em)
print("SOM ARI:", ari_som)



# ----------------
# save the labels
# ----------------
df["Gas_Quality"].to_csv("quality_labels.csv", index=False)
