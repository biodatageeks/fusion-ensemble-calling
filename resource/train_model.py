import pandas as pd
from pycaret.classification import *
from pathlib import Path

# --- Define paths relative to the repository root ---
repo_root = Path(__file__).parent.parent  # source/ -> ../
data_dir = repo_root / "data"
models_dir = repo_root / "models"
models_dir.mkdir(exist_ok=True)  # create models dir if it doesn't exist

data_file = data_dir / "merged_restructured.csv"
model_file = models_dir / "fusion_detection_model"

# --- Load data ---
df = pd.read_csv(data_file)

# --- Filter out breast cancer samples ---
df = df[df["Meta1_(cancer_type)"].str.lower() != "breast"]

# --- Select only specific tools ---
selected_tools = ["ARRIBA", "FUSIONCATCHER_v1.10_June192019", "STAR_FUSION_v1.5"]
detection_cols = [f"{tool}-detection" for tool in selected_tools]

# --- Build dataset with features and label ---
data = df[detection_cols + ['Label (fusion_correct)']]

# --- Initialize PyCaret environment ---
clf = setup(
    data=data,
    target='Label (fusion_correct)',
    train_size=0.8,
    session_id=123,
    verbose=False,
    html=False,
    use_gpu=True
)

# --- Compare models ---
best_model = compare_models()

# --- Train the best model ---
final_model = finalize_model(best_model)

# --- Save the model in models directory ---
save_model(final_model, str(model_file))

# --- Example predictions ---
predictions = predict_model(final_model, data=data)
print(predictions.head())
