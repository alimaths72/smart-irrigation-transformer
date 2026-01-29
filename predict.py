# -------------------------------
# Fix OpenMP runtime issue (Windows)
# -------------------------------
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import joblib
import numpy as np

# -------------------------------
# Transformer model definition
# (must match training exactly)
# -------------------------------
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# -------------------------------
# Load trained model & scaler
# -------------------------------
model = SimpleTransformer(input_dim=3, num_classes=3)
model.load_state_dict(torch.load("models/transformer_model.pth"))
model.eval()

scaler = joblib.load("models/scaler.pkl")

# -------------------------------
# User input for prediction
# -------------------------------
print("\n--- Irrigation Requirement Prediction ---")
print("Enter values within these ranges:")
print("• Moisture Index (MOI): (1--100) ")
print("• Temperature (°C): (1--50) ")
print("• Humidity (%): (1--100) ")

try:
    moi = float(input("MOI: "))
    temp = float(input("Temperature (°C): "))
    humidity = float(input("Humidity (%): "))

    user_data = np.array([[moi, temp, humidity]])
    user_data_scaled = scaler.transform(user_data)

    user_tensor = torch.tensor(user_data_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(user_tensor)
        prediction = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, prediction].item()

    if prediction == 1:
        print("\nPrediction: Irrigation LIKELY REQUIRED")
    else:
        print("\nPrediction: Irrigation LIKELY NOT required")

    print(f"Prediction confidence: {confidence:.2f}")

except ValueError:
    print("Invalid input. Please enter numeric values only.")
