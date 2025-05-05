import pandas as pd

# Load both CSV files
algo = "diffusion_map"
df1 = pd.read_csv(f"../old/{algo}_kmeans.csv")
df2 = pd.read_csv(f"../old1/{algo}_kmeans.csv")

# Specify the column to compare and the selection rule
compare_column = "adjusted_rand_score"  # Change this as needed
selection_rule = "max"  # Use "max" or "min"

# Prepare a new DataFrame to store the results
merged_rows = []

for i in range(len(df1)):
    row1 = df1.iloc[i]
    row2 = df2.iloc[i]

    if selection_rule == "max":
        selected_row = row1 if row1[compare_column] >= row2[compare_column] else row2
    else:
        selected_row = row1 if row1[compare_column] <= row2[compare_column] else row2

    merged_rows.append(selected_row)

# Convert list of Series to DataFrame
merged_df = pd.DataFrame(merged_rows)

# Save to a new CSV
merged_df.to_csv(f"../{algo}_kmeans.csv", index=False)