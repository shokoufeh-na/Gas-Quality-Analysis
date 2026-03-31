import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
X = pd.read_csv("StdGasProperties.csv").values
labels_df = pd.read_csv("quality_labels.csv")
y = labels_df["Gas_Quality"].map({"Regular":0, "Medium":1, "Premium":2}).values

# ---------------------------------------------------------
# 2. Train/Val/Test Split (70/15/15)
# ---------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ---------------------------------------------------------
# 3. Standardize (fit on training only)
# ---------------------------------------------------------
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val   = scaler.transform(X_val)
# X_test  = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. K-Means to get RBF centers
# ---------------------------------------------------------
num_centers = 20   # number of hidden units
kmeans = KMeans(n_clusters=num_centers, random_state=42)
kmeans.fit(X_train)
centers = kmeans.cluster_centers_

# ---------------------------------------------------------
# 5. Compute kernel width σ
#    Use average distance between centers
# ---------------------------------------------------------
dists = []
for i in range(num_centers):
    for j in range(i+1, num_centers):
        dists.append(np.linalg.norm(centers[i] - centers[j]))

sigma = np.mean(dists)

# ---------------------------------------------------------
# 6. RBF transformation
# ---------------------------------------------------------
def rbf_transform(X, centers, sigma):
    Phi = np.zeros((X.shape[0], len(centers)))
    for i, c in enumerate(centers):
        Phi[:, i] = np.exp(-np.linalg.norm(X - c, axis=1)**2 / (2 * sigma**2))
    return Phi

Phi_train = rbf_transform(X_train, centers, sigma)
Phi_test  = rbf_transform(X_test, centers, sigma)

# ---------------------------------------------------------
# 7. Train output weights using least squares
# ---------------------------------------------------------
# One-hot encode labels
num_classes = 3
Y_train = np.eye(num_classes)[y_train]

# Solve for weights: W = (Phiᵀ Phi)^(-1) Phiᵀ Y
W = np.linalg.pinv(Phi_train) @ Y_train

# ---------------------------------------------------------
# 8. Predict on test set
# ---------------------------------------------------------
scores = Phi_test @ W
y_pred = np.argmax(scores, axis=1)

# ---------------------------------------------------------
# 9. Evaluation
# ---------------------------------------------------------
conf_mat = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== RBF Classifier Results ===")
print(f"Number of hidden units: {num_centers}")
print(f"Kernel width (sigma): {sigma:.4f}")

print("\n=== Confusion Matrix ===")
print(conf_mat)

print("\n=== Test Accuracy ===")
print(f"{acc*100:.2f}%")

print("\n=== F1 Score (Macro) ===")
print(f"{f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Regular", "Medium", "Premium"]))
