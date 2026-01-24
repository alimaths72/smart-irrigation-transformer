import os
import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------
# Load dataset safely
# --------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cropdata_updated.csv")

df = pd.read_csv(DATA_PATH)

# --------------------------------
# Select features and target
# --------------------------------

X = df[["MOI", "temp", "humidity"]]
y = df["result"]

# --------------------------------
# Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# Print shapes to verify split
# --------------------------------

print("Training set size:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest set size:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
