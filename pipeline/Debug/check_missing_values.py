import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv("D:/facemesh/facemesh_coordinates.csv")

# Check for missing values in the landmark columns
missing_values = df.isnull().sum()
missing_landmarks = missing_values[missing_values > 0]

# Print the missing values
if not missing_landmarks.empty:
    print("Missing values found in the following landmark columns:")
    print(missing_landmarks)
else:
    print("No missing values found in the landmark columns.")
