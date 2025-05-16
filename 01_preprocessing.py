import pandas as pd

# Load dataset
df = pd.read_csv("telco_churn.csv")

# Drop unnecessary columns
columns_to_drop = [
    'Customer ID', 'Country', 'State', 'City', 'Zip Code',
    'Latitude', 'Longitude', 'Churn Score', 'Churn Category',
    'Churn Reason', 'Customer Status'
]
df.drop(columns=columns_to_drop, axis=1, inplace=True, errors='ignore')

# Convert 'Total Charges' to numeric
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Drop rows with missing values
df.dropna(inplace=True)

# Convert target column 'Churn Label' to new binary column 'Churn'
if 'Churn Label' in df.columns:
    df['Churn'] = df['Churn Label'].map({'Yes': 1, 'No': 0})
    df.drop('Churn Label', axis=1, inplace=True)

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)

# Save cleaned dataset
df.to_csv("clean_telco_churn.csv", index=False)
print("Cleaned dataset saved as clean_telco_churn.csv")
