import pandas as pd
from pathlib import Path

# Define paths
data_dir = Path("data")
input_excel = data_dir / "13059_2019_1842_MOESM1_ESM.xlsx"
truthset_file = data_dir / "truthset_min7.csv"

merged_output = data_dir / "merged_output.csv"
merged_with_truth = data_dir / "merged_with_truth.csv"
merged_restructured = data_dir / "merged_restructured.csv"

# --- Step 1: Merge sheets S4 and S9 from Excel ---
print("Step 1: Merging Table S4 and Table S9...")

sheet1 = "Table S4 cancer cell line preds"
sheet2 = "Table S9 - cancer cell lines"

df_preds = pd.read_excel(input_excel, sheet_name=sheet1)
df_lines = pd.read_excel(input_excel, sheet_name=sheet2)

merged_df = df_preds.merge(
    df_lines[["Cancer cell line", "Cancer type"]],
    left_on="sample",
    right_on="Cancer cell line",
    how="left"
)

merged_df.drop(columns=["Cancer cell line"], inplace=True)

# Move 'Cancer type' to the second position
cols = list(merged_df.columns)
cols.insert(1, cols.pop(cols.index("Cancer type")))
merged_df = merged_df[cols]

# Sort by 'fusion'
merged_df = merged_df.sort_values(by="fusion", ascending=True)
merged_df.to_csv(merged_output, index=False)

print(f"Saved: {merged_output}")

# --- Step 2: Merge with truthset ---
print("Step 2: Adding truthset labels...")

df_truth = pd.read_csv(truthset_file)
df_merged = pd.read_csv(merged_output)

truth_pairs = set(df_truth["sample_fusion"])

df_merged["fusion_correct"] = df_merged.apply(
    lambda row: 1 if f"{row['sample']}|{row['fusion']}" in truth_pairs else 0,
    axis=1
)

# Move 'fusion_correct' to 5th position
cols = list(df_merged.columns)
cols.insert(4, cols.pop(cols.index("fusion_correct")))
df_merged = df_merged[cols]

df_merged.to_csv(merged_with_truth, index=False)
print(f"Saved: {merged_with_truth}")

# --- Step 3: Restructure merged data ---
print("Step 3: Restructuring dataset...")

tools = df_merged["prog"].unique()

def build_row(group):
    row = {}
    row["Sample-fusion"] = f"{group['sample'].iloc[0]}-{group['fusion'].iloc[0]}"
    row["Meta1_(cancer_type)"] = group["Cancer type"].iloc[0]
    row["Label (fusion_correct)"] = group["fusion_correct"].iloc[0]
    
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

grouped = df_merged.groupby(["sample", "fusion"])
rows = [build_row(group) for _, group in grouped]

df_out = pd.DataFrame(rows)
df_out.to_csv(merged_restructured, index=False)

print(f"Final dataset saved as: {merged_restructured}")
