import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)

# Load production data
prod_df = pd.read_csv("data/production_data.csv")
# load the Model 
model=joblib.load(
    'model/churn_model.pkl'
)
# lpoad production data
X_prod=prod_df.drop(
    ['CustomerID','Churn'],axis=1
)

# actual values 
y_true=prod_df['Churn'].apply(lambda x: 1 if x=="Yes" else 0)

# prediction 
y_pred=model.predict(X_prod)

# claculate metrics 
accuracy=accuracy_score(y_true,y_pred)
precision=precision_score(y_true,y_pred,zero_division=0
)

recall=recall_score(y_true,y_pred,zero_division=0)
f1=f1_score(y_true,y_pred,zero_division=0)


# diaplay the metrics
print("Model Performance monitoring")
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall : {recall:.2f}")
print(f"F1-score :{f1:.2f}")


# Alert
if accuracy <0.70:
    print("Alert: Model Performance Degraded!")

else:
    print("Model Performing Well.")