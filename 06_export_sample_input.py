import pandas as pd
import os

# Load dataset
df = pd.read_csv("clean_telco_churn.csv")
X = df.drop("Churn", axis=1)
sample = X.iloc[[0]]

# Export to JSON
sample.to_json("sample_input.json", orient="split", indent=2)

# Print debug info
print(" Exported sample_input.json")
print(" Location:", os.getcwd())
print(" File exists:", os.path.exists("sample_input.json"))
