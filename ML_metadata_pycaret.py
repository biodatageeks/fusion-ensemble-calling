import pandas as pd
from pycaret.classification import *
# =============================
# 1. Wczytanie danych
# =============================
df = pd.read_csv("merged_restructured.csv")

# =============================
# 2. Wybór kolumn cech
# =============================
detection_cols = [col for col in df.columns if col.endswith("-detection")]
j_cols         = [col for col in df.columns if col.endswith("-J")]
s_cols         = [col for col in df.columns if col.endswith("-S")]
meta_cols      = ["Meta1_(cancer_type)"]
target_col     = "Label (fusion_correct)"

features = detection_cols + j_cols + s_cols + meta_cols
data = df[features + [target_col]]

# =============================
# 3. Inicjalizacja PyCaret
# =============================
clf = setup(
    data=data,
    target=target_col,
    session_id=123,
    train_size=0.8,
    categorical_features=meta_cols,
    verbose=False,
    html=False
)

# =============================
# 4. Porównanie modeli
# =============================
best_model = compare_models()

# =============================
# 5. Finalizacja najlepszego modelu
# =============================
final_model = finalize_model(best_model)

# =============================
# 6. Predykcje na całym zbiorze
# =============================
preds = predict_model(final_model, data=data)
print(preds.head())

# zapis predykcji do pliku CSV
preds.to_csv("fusion_predictions.csv", index=False)
print("Predykcje zapisane jako 'fusion_predictions.csv'")

# =============================
# 7. Zapisanie modelu
# =============================
save_model(final_model, "fusion_model")
print("Model zapisany jako 'fusion_model.pkl'")

