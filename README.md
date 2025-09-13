# fusion-ensemble-calling
# RNA Fusion Detection Pipeline

This repository contains scripts and data for detecting gene fusions from RNA sequencing results. The workflow is modular and consists of the following key components:

## 1. Data Preparation
- Merge raw Excel tables (Table S4 and Table S9) containing fusion predictions and cancer cell line information.  
- Integrate a truth set to label correct fusions.  
- Restructure the dataset for machine learning.  

## 2. Machine Learning Model
- Only selected tools are considered: ARRIBA, FUSIONCATCHER_v1.10_June192019, STAR_FUSION_v1.5.  
- Samples from breast cancer are excluded.  
- Features include detection flags, J and S counts, and cancer type.  
- Models are trained and compared using PyCaret, with the best model saved in `models/` and predictions saved in `data/`.  

## 3. Repository Structure
- `data/` – raw and processed data, including merged and restructured datasets.  
- `models/` – trained machine learning models.  
- `source/` – Python scripts for dataset preparation and model training.  

## 4. Usage
- Run dataset preparation scripts to generate a final structured dataset.  
- Train the classifier on the structured dataset, save predictions, and store the trained model.  
- All paths are relative to the repository root for reproducibility.  

## 5. Reproducibility and Modularity
- The pipeline allows easy updates to input data, features, or models without changing the overall structure.  

## 6. References
- Original dataset and supplementary tables: [Genome Biology](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1842-9#Sec19)  
- PyCaret documentation: [https://pycaret.]()
