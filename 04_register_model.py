import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("clean_telco_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define best model from tuning
best_params = {'n_estimators': 110, 'max_depth': 9, 'min_samples_split': 8}

with mlflow.start_run(run_name="final_rf_model"):
    clf = RandomForestClassifier(**best_params, random_state=42)
    clf.fit(X_train, y_train)

    # Log and register the model
    mlflow.sklearn.log_model(clf, "model", registered_model_name="TelcoChurnRF")
    print("✅ Final model trained and registered!")
