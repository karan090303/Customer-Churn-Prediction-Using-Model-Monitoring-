import streamlit as st
import pandas as pd
import os
import joblib

from sklearn.metrics import accuracy_score


st.set_page_config(
    page_title="ML Model Monitoring",
    page_icon="",
    layout="wide"
)


st.title(
    " Customer Churn Model Monitoring Dashboard"
)

st.write(
    "Monitor model performance and production data drift."
)


# ------------------------------------------------
# Load Model
# ------------------------------------------------

model_path = "model/churn_model.pkl"

if os.path.exists(model_path):

    model = joblib.load(model_path)

else:

    st.error("Model not found!")
    st.stop()


# ------------------------------------------------
# Load Production Data
# ------------------------------------------------

prod_df = pd.read_csv(
    "data/production_data.csv"
)


X_prod = prod_df.drop(
    ["CustomerID", "Churn"],
    axis=1
)

y_true = prod_df["Churn"].apply(
    lambda x: 1 if x == "Yes" else 0
)


# ------------------------------------------------
# Prediction
# ------------------------------------------------

y_pred = model.predict(X_prod)


accuracy = accuracy_score(
    y_true,
    y_pred
)


# ------------------------------------------------
# Dashboard Metrics
# ------------------------------------------------

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with col2:

    if accuracy < 0.70:
        status = "ALERT"
    else:
        status = "Healthy"

    st.metric(
        "Model Status",
        status
    )


with col3:

    st.metric(
        "Production Records",
        len(prod_df)
    )


# ------------------------------------------------
# Production Data
# ------------------------------------------------

st.subheader(
    "Production Data"
)

st.dataframe(
    prod_df,
    use_container_width=True,
    width="stretch"
)


# ------------------------------------------------
# Prediction Distribution
# ------------------------------------------------

st.subheader(
    "Prediction Distribution"
)

prediction_counts = pd.Series(
    y_pred
).value_counts()

prediction_counts.index = [
    "No Churn" if x == 0 else "Churn"
    for x in prediction_counts.index
]

st.bar_chart(
    prediction_counts
)

# ------------------------------------------------
# Drift Report
# ------------------------------------------------

st.subheader(
    "Data Drift Report"
)

report_path = (
    "monitoring/data_drift_report.html"
)


if os.path.exists(report_path):

    st.success(
        "Data drift report generated successfully."
    )

    with open(
        report_path,
        "r",
        encoding="utf-8"
    ) as file:

        html = file.read()

    st.download_button(
        label="Download Drift Report",
        data=html,
        file_name="data_drift_report.html",
        mime="text/html"
    )

else:

    st.warning(
        "Drift report not found. "
        "Run data_drift.py first."
    )


# ------------------------------------------------
# Retraining Recommendation
# ------------------------------------------------

st.subheader(
    "Recommended Action"
)

if accuracy < 0.70:

    st.error(
        "Model performance degraded. "
        "Retraining recommended."
    )

else:

    st.success(
        "Model is performing within the "
        "acceptable threshold."
    )