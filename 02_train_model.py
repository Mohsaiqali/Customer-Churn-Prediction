import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Enable MLflow auto-logging
mlflow.sklearn.autolog()

# Load the preprocessed dataset
df = pd.read_csv("clean_telco_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
with mlflow.start_run():

    # Train model
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    # Log custom metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    print(f"Model trained and logged: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}")
