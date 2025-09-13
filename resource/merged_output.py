import pandas as pd
from pathlib import Path

# Define paths relative to the repository root
data_dir = Path("data")
input_file = data_dir / "13059_2019_1842_MOESM1_ESM.xlsx"
output_file = data_dir / "merged_output.csv"

# Correct sheet names from the Excel file
sheet1 = "Table S4 cancer cell line preds"
sheet2 = "Table S9 - cancer cell lines"

# Load sheets from Excel file
df_preds = pd.read_excel(input_file, sheet_name=sheet1)
df_lines = pd.read_excel(input_file, sheet_name=sheet2)

# Merge based on 'sample' (from df_preds) and 'Cancer cell line' (from df_lines)
merged_df = df_preds.merge(
    df_lines[["Cancer cell line", "Cancer type"]],
    left_on="sample",
    right_on="Cancer cell line",
    how="left"
)

# Drop duplicate 'Cancer cell line' column after merging
merged_df.drop(columns=["Cancer cell line"], inplace=True)

# Move 'Cancer type' column to the second position
cols = list(merged_df.columns)
cols.insert(1, cols.pop(cols.index("Cancer type")))
merged_df = merged_df[cols]

# Sort rows alphabetically by 'fusion'
merged_df = merged_df.sort_values(by="fusion", ascending=True)

# Save the merged DataFrame to CSV
merged_df.to_csv(output_file, index=False)

print(f"Merged file saved as: {output_file}")
