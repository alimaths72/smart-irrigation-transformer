import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

X = df[["MOI", "temp", "humidity"]].values
y = df["result"].values

# --------------------------------
# Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# Feature scaling (VERY IMPORTANT)
# --------------------------------

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# --------------------------------
# Simple Transformer Model
# --------------------------------

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, 16)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        # x shape: (batch_size, features)
        x = x.unsqueeze(1)        # (batch_size, 1, features)
        x = self.embedding(x)     # (batch_size, 1, 16)
        x = self.transformer(x)   # (batch_size, 1, 16)
        x = x.mean(dim=1)         # (batch_size, 16)
        x = self.fc(x)            # (batch_size, num_classes)
        return x

# --------------------------------
# Model setup
# --------------------------------

num_classes = len(set(y))
model = SimpleTransformer(input_dim=3, num_classes=num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------
# Training loop
# --------------------------------

epochs = 10

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# --------------------------------
# Evaluation
# --------------------------------

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = accuracy_score(y_test, predictions)

print("Transformer Model Accuracy:", accuracy)
