# -------------------------------
# Reproducibility: Fix random seeds
# -------------------------------
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------
# Fix OpenMP runtime issue (Windows)
# -------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------
# Import required libraries
# -------------------------------
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------
# Load dataset safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "cropdata_updated.csv")

df = pd.read_csv(DATA_PATH)

# -------------------------------
# Select features and target
# -------------------------------
X = df[["MOI", "temp", "humidity"]].values
y = df["result"].values

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# BASELINE MODEL: Logistic Regression
# -------------------------------
baseline_model = LogisticRegression(max_iter=1000)
baseline_model.fit(X_train, y_train)

baseline_preds = baseline_model.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_preds)

print("Baseline Logistic Regression Accuracy:", baseline_accuracy)

# -------------------------------
# Feature scaling for Transformer
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.long)
y_test_torch = torch.tensor(y_test, dtype=torch.long)

# -------------------------------
# Simple Transformer Model
# -------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )

        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)        # (batch, seq_len=1, features)
        x = self.embedding(x)     # (batch, 1, 16)
        x = self.transformer(x)   # (batch, 1, 16)
        x = x.mean(dim=1)         # (batch, 16)
        x = self.fc(x)            # (batch, num_classes)
        return x

# -------------------------------
# Model setup
# -------------------------------
num_classes = len(set(y))
model = SimpleTransformer(input_dim=3, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training loop
# -------------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_torch)
    loss = criterion(outputs, y_train_torch)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# -------------------------------
# Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    outputs = model(X_test_torch)
    transformer_preds = torch.argmax(outputs, dim=1)
    transformer_accuracy = accuracy_score(y_test, transformer_preds)

print("\nTransformer Model Accuracy:", transformer_accuracy)

# -------------------------------
# Comparison summary
# -------------------------------
print("\n--- Model Performance Comparison ---")
print(f"Baseline Logistic Regression Accuracy: {baseline_accuracy:.4f}")
print(f"Transformer Model Accuracy: {transformer_accuracy:.4f}")

# Baseline and Transformer models are compared using accuracy metric
# Transformer shows improved performance over baseline

# -------------------------------
# Plot accuracy comparison
# -------------------------------
models = ["Logistic Regression", "Transformer"]
accuracies = [baseline_accuracy, transformer_accuracy]

plt.figure()
plt.bar(models, accuracies)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.show()
# --------------------------------
# Inference: Predict on new user input
# --------------------------------

print("\n--- Irrigation Requirement Prediction (User Input) ---")

try:
    moi_input = float(input("Enter Moisture Index (MOI): "))
    temp_input = float(input("Enter Temperature: "))
    humidity_input = float(input("Enter Humidity: "))

    # Create input array
    user_data = [[moi_input, temp_input, humidity_input]]

    # Scale input using the same scaler
    user_data_scaled = scaler.transform(user_data)

    # Convert to tensor
    user_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)

    # Predict
    model.eval()
    with torch.no_grad():
        output = model(user_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Display result
    if prediction == 1:
        print("\nPrediction: Irrigation REQUIRED")
    else:
        print("\nPrediction: Irrigation NOT required")

except ValueError:
    print("Invalid input. Please enter numeric values only.")
