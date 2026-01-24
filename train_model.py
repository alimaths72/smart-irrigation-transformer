import os
import pandas as pd

# Find the folder where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "cropdata_updated.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Show column names
print("Columns in dataset:")
print(df.columns)

# Show basic information
print("\nDataset info:")
print(df.info())
