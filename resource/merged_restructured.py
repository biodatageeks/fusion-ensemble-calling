import pandas as pd
from pathlib import Path

# Define paths relative to the repository root
data_dir = Path("data")
input_file = data_dir / "merged_with_truth.csv"
output_file = data_dir / "merged_restructured.csv"

# Load merged dataset with truthset
df = pd.read_csv(input_file)

# List all tools from the 'prog' column
tools = df["prog"].unique()

# Function to build a row for each sample-fusion pair
def build_row(group):
    row = {}
    row["Sample-fusion"] = f"{group['sample'].iloc[0]}-{group['fusion'].iloc[0]}"
    row["Meta1_(cancer_type)"] = group["Cancer type"].iloc[0]
    row["Label_(fusion_correct)"] = group["fusion_correct"].iloc[0]
    
    for tool in tools:
        subset = group[group["prog"] == tool]
        if not subset.empty:
            row[f"{tool}-detection"] = 1
            row[f"{tool}-J"] = subset["J"].values[0]
            row[f"{tool}-S"] = subset["S"].values[0]
        else:
            row[f"{tool}-detection"] = 0
            row[f"{tool}-J"] = 0
            row[f"{tool}-S"] = 0
    return row

# Group by sample and fusion
grouped = df.groupby(["sample", "fusion"])

# Build new rows
rows = [build_row(group) for _, group in grouped]

# Create the new DataFrame and save
df_out = pd.DataFrame(rows)
df_out.to_csv(output_file, index=False)

print(f"Restructured file saved as: {output_file}")
