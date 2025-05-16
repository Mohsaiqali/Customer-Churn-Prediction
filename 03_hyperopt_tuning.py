import pandas as pd
import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
df = pd.read_csv("clean_telco_churn.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(params):
    with mlflow.start_run(nested=True):
        clf = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            random_state=42
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log hyperparameters and metric
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        return {'loss': -acc, 'status': STATUS_OK}

# Define the search space
search_space = {
    'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# Track tuning under one main experiment
with mlflow.start_run(run_name="hyperopt_tuning_rf"):
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    print(f"Best hyperparameters: {best_result}")
