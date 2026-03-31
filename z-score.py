import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset
df = pd.read_csv("GasProperties.csv")

# 2. Select the four chemical properties
features = ["T", "P", "TC", "SV"]
X = df[features]

# 3. Print mean and std BEFORE standardization
print("Before standardization:")
print(X.mean())
print(X.std())

# 4. Apply z-score scaling
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Convert back to DataFrame
df_std = pd.DataFrame(X_std, columns=features)

# 5. Save to StdGasProperties.csv
df_std.to_csv("StdGasProperties.csv", index=False)

# 6. Print mean and std AFTER standardization
print("\nAfter standardization:")
print(df_std.mean())
print(df_std.std())
