import pandas as pd
from pycaret.classification import *

# Wczytaj dane
df = pd.read_csv("/data/nextflow_work/rnafusion/ML/merged_restructured.csv")

# Wybieramy tylko kolumny z "-detection" jako cechy
detection_cols = [col for col in df.columns if col.endswith("-detection")]

# Budujemy nową ramkę danych z cechami i etykietą
data = df[detection_cols + ['Label (fusion_correct)']]

# Inicjalizacja środowiska PyCaret
clf = setup(
    data=data,
    target='Label (fusion_correct)',
    train_size=0.8,
    session_id=123,
    verbose=False,
    html=False,
    use_gpu=True
)


# Porównanie modeli
best_model = compare_models()

# Wytrenuj najlepszy model
final_model = finalize_model(best_model)

# Zapisz model
save_model(final_model, "fusion_detection_model")

# Przykładowe predykcje
predictions = predict_model(final_model, data=data)
print(predictions.head())
