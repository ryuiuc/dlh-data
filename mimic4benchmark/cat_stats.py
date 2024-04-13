import pandas as pd
import os
import glob

# Path pattern to match all timeseries CSV files in the current directory
file_pattern = "*timeseries.csv"

# Container for the unique values in the non-numeric columns of each file
unique_values = {}

# Iterate over all files that match the pattern
for file_name in glob.glob(file_pattern):
    # Read the CSV file
    df = pd.read_csv(file_name)

    # Filter columns to include only non-numeric types
    non_numeric_columns = df.select_dtypes(exclude=['number']).columns

    # Process each row in the DataFrame
    for _, row in df.iterrows():
        for col_name in non_numeric_columns:
            value = row[col_name]
            if not pd.isna(value):
                if f"{col_name}" not in unique_values:
                    unique_values[f"{col_name}"] = set()
                unique_values[f"{col_name}"].add(value)

# Convert the sets to lists for easier viewing
for key in unique_values.keys():
    unique_values[key] = list(unique_values[key])

print(unique_values)
