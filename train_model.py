import os
import pandas as pd

# Get absolute path of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "cropdata_updated.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# -------------------------------
# Select features and target
# -------------------------------

# Features (inputs)
X = df[["MOI", "temp", "humidity"]]

# Target (output)
y = df["result"]

# Print shapes to confirm separation
print("Feature matrix shape (X):", X.shape)
print("Target vector shape (y):", y.shape)

# Display first few rows
print("\nFirst 5 rows of features:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())
