import pandas as pd
from pathlib import Path

# Define paths relative to the repository root
data_dir = Path("data")
merged_path = data_dir / "merged_output.csv"
truthset_path = data_dir / "truthset_min7.csv"
output_path = data_dir / "merged_with_truth.csv"

# Load data
df_merged = pd.read_csv(merged_path)
df_truth = pd.read_csv(truthset_path)

# Create a set of "sample|fusion" pairs from the truthset
truth_pairs = set(df_truth["sample_fusion"])

# Add a column 'fusion_correct' (1 if sample|fusion pair is in the truthset, else 0)
df_merged["fusion_correct"] = df_merged.apply(
    lambda row: 1 if f"{row['sample']}|{row['fusion']}" in truth_pairs else 0,
    axis=1
)

# Move 'fusion_correct' to the 5th position
cols = list(df_merged.columns)
cols.insert(4, cols.pop(cols.index("fusion_correct")))
df_merged = df_merged[cols]

# Save the result to a new CSV file
df_merged.to_csv(output_path, index=False)

print(f"Merged file with truthset saved as: {output_path}")
