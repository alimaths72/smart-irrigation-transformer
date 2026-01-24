import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# Train baseline model
# --------------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# --------------------------------
# Evaluate model
# --------------------------------

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Baseline Logistic Regression Accuracy:", accuracy)
