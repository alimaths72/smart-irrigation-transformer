import os
import pandas as pd

# Get the directory where this Python file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the full path to the dataset
DATA_PATH = os.path.join(BASE_DIR, "data", "cropdata_updated.csv")

# Load the dataset
df = pd.read_csv(DATA_PATH)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)
