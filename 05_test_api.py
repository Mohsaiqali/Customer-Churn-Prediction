import requests
import json

# Load sample from file
with open("sample_input.json", "r") as f:
    split_data = json.load(f)

# Wrap in MLflow 2.0+ format
payload = {
    "dataframe_split": split_data
}

# Send request
url = "http://127.0.0.1:1234/invocations"
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json.dumps(payload))

# Output result
print("Prediction result:", response.json())
