## 📌 Project Overview

This project demonstrates a complete machine learning lifecycle management system using **MLflow**. The goal is to predict customer churn in the telecom industry and manage every step from data preprocessing to deployment and model registry.

---

## 📁 Project Structure

. ├── 01_preprocessing.py # Data cleaning and encoding ├── 02_train_model.py # Initial model training & tracking ├── 03_hyperopt_tuning.py # Hyperparameter tuning using Hyperopt ├── 04_register_model.py # Model registration in MLflow ├── 05_test_api.py # REST API prediction test ├── 06_export_sample_input.py # Exports a sample input as JSON ├── clean_telco_churn.csv # Cleaned dataset ├── sample_input.json # Real input used for API ├── requirements.txt # Python dependencies └── README.md # This file

---

## 🧠 ML Model

- **Model:** RandomForestClassifier
- **Tuned With:** Hyperopt
- **Tracking:** MLflow
- **Deployment:** Local REST API via `mlflow models serve`
- **Registry:** Model versioning & staging lifecycle managed via MLflow UI

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt


Preprocess the data:
python 01_preprocessing.py

Train and tune the model:
python 02_train_model.py
python 03_hyperopt_tuning.py

Register and serve:
python 04_register_model.py
mlflow models serve -m "models:/TelcoChurnRF/1" -p 1234 --no-conda

Predict with API:
python 06_export_sample_input.py
python 05_test_api.py


✅ Results
Accuracy: ~96–97%
Model served and tested via REST API
Model version tracked and promoted in MLflow
