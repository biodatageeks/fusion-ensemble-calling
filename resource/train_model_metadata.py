import pandas as pd
from pycaret.classification import *
from pathlib import Path

# =============================
# 0. Define paths
# =============================
repo_root = Path(__file__).parent.parent
data_dir = repo_root / "data"
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)

input_file = data_dir / "merged_restructured.csv"
pred_file = data_dir / "fusion_predictions.csv"
model_file = models_dir / "fusion_model"

# =============================
# 1. Load data
# =============================
df = pd.read_csv(input_file)

# =============================
# 2. Filter data
# =============================
df = df[df["Meta1_(cancer_type)"].str.lower() != "breast"]

# =============================
# 3. Select features for 3 tools
# =============================
selected_tools = ["ARRIBA", "FUSIONCATCHER_v1.10_June192019", "STAR_FUSION_v1.5"]

detection_cols = [f"{tool}-detection" for tool in selected_tools]
j_cols = [f"{tool}-J" for tool in selected_tools]
s_cols = [f"{tool}-S" for tool in selected_tools]
meta_cols = ["Meta1_(cancer_type)"]
target_col = "Label (fusion_correct)"

features = detection_cols + j_cols + s_cols + meta_cols
data = df[features + [target_col]]

# =============================
# 4. Initialize PyCaret
# =============================
clf = setup(
    data=data,
    target=target_col,
    session_id=123,
    train_size=0.8,
    categorical_features=meta_cols,
    verbose=False,
    html=False,
    use_gpu=True
)

# =============================
# 5. Compare models
# =============================
best_model = compare_models()

# =============================
# 6. Finalize best model
# =============================
final_model = finalize_model(best_model)

# =============================
# 7. Predict on full dataset
# =============================
preds = predict_model(final_model, data=data)
print(preds.head())

# Save predictions
preds.to_csv(pred_file, index=False)
print(f"Predictions saved as '{pred_file}'")

# =============================
# 8. Save trained model
# =============================
save_model(final_model, str(model_file))
print(f"Model saved as '{model_file}.pkl'")
