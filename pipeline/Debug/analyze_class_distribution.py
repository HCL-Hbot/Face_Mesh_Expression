import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("D:/facemesh/facemesh_coordinates.csv")

# Print the total number of samples
print(f"Total samples in dataframe: {len(df)}")

# Print the class distribution
print("\nClass distribution:")
print(df["label"].value_counts())
