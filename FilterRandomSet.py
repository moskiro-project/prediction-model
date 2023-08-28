import pandas as pd

# Read the original Excel file
df = pd.read_excel('UncleanedFullDataset.xlsx')

# Get the total number of rows in the file
total_rows = df.shape[0]

# Set the number of random rows to select
num_random_rows = 500

# Generate random row indices
random_indices = pd.Series(pd.Index(range(total_rows))).sample(num_random_rows)

# Select the random rows
random_rows = df.loc[random_indices]

# Write the random rows to a new Excel file
random_rows.to_excel('ReducedDataset500Entries.xlsx', index=False)