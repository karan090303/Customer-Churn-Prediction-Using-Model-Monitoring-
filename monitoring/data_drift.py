import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

# Load training data
train_df = pd.read_csv("data/train_data.csv")

# Load production data
prod_df = pd.read_csv("data/production_data.csv")

# Remove ID and target columns

train_features = train_df.drop(
    ["CustomerID", "Churn"], axis=1
)

prod_features = prod_df.drop(
    ["CustomerID", "Churn"], axis=1
)

print("Training Features Shape:", train_features.shape)
print("Production Features Shape:", prod_features.shape)

# Create Evidently report
report = Report([
    DataDriftPreset()
])

# Run report
snapshot = report.run(
    reference_data=train_features,
    current_data=prod_features
)

# Save HTML report
snapshot.save_html("monitoring/data_drift_report.html")

print("Data Drift report generated successfully!")
print("Open: monitoring/data_drift_report.html")