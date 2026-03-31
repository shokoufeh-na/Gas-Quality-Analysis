import pandas as pd
from sklearn.mixture import GaussianMixture
import numpy as np

# Load standardized data
df = pd.read_csv("StdGasProperties.csv")

# Use df directly (it is already standardized)
X = df.values
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',
    init_params='kmeans',
    tol=1e-3,
    # max_iter=100,
    random_state=42
)

gmm.fit(X)

print("Initialization method:", gmm.init_params)
print("Tolerance:", gmm.tol)
# print("Maximum iterations:", gmm.max_iter)

print("Covariance type:", gmm.covariance_type)

# Cluster labels for each sample
em_labels = gmm.predict(X)

# Print parameters for each cluster
for k in range(gmm.n_components):

    print(f"\nCluster {k}")
    print("----------------------")

    print(f"Mixture weight (π_{k}):")
    print(gmm.weights_[k])

    print(f"\nMean vector (μ_{k}):")
    print(gmm.means_[k])

    print(f"\nCovariance matrix (Σ_{k}):")
    print(gmm.covariances_[k])

print("\nConverged:", gmm.converged_)
print("Iterations:", gmm.n_iter_)

probs = gmm.predict_proba(X)

print("Posterior probabilities p(z=k | x) for first 3 samples:\n")

for i in range(3):
    print(f"Sample {i+1}:")
    for k in range(3):
        print(f"  p(z={k} | x) = {probs[i][k]:.4f}")
    print()

# Save EM labels for wobbe-index.py
np.savetxt("em_labels.csv", em_labels, fmt="%d")

# -------------------------------
#  Calculate silhouette score
# -------------------------------

from sklearn.metrics import silhouette_score

# Hard cluster assignments for EM/GMM
gmm_labels = gmm.predict(X)

# Silhouette score
em_silhouette = silhouette_score(X, gmm_labels)

print("Silhouette Score (EM/GMM):", em_silhouette)