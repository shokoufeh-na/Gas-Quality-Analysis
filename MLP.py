import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("StdGasProperties.csv")

# Features (T, P, TC, SV)
X = df.values

# True labels (Regular, Medium, Premium)

labels_df = pd.read_csv("quality_labels.csv")
y = labels_df["Gas_Quality"].map({"Regular":0, "Medium":1, "Premium":2}).values

# ---------------------------------------------------------
# 2. Train/Validation/Test Split (70/15/15)
# ---------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ---------------------------------------------------------
# 3. Standardization (fit on training only)
# ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ---------------------------------------------------------
# 4. MLP Classifier Configuration
# ---------------------------------------------------------
mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),   # Topology
    activation='relu',
    solver='adam',                # Optimizer
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=32,
    alpha=0.0001,                 # L2 regularization
    random_state=42
)

# ---------------------------------------------------------
# 5. Train the model
# ---------------------------------------------------------
mlp.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Evaluate on test set
# ---------------------------------------------------------
y_pred = mlp.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== Confusion Matrix ===")
print(conf_mat)

print("\n=== Test Accuracy ===")
print(f"{acc*100:.2f}%")

print("\n=== F1 Score (Macro) ===")
print(f"{f1:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["Regular", "Medium", "Premium"]))
