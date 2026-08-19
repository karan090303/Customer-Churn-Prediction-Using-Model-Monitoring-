import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Show all columns when displaying DataFrame
pd.set_option('display.max_columns', None)
# load the training data 
df=pd.read_csv("data/train_data.csv")

print("Training data loaded")
print(df.head())

X=df.drop(['CustomerID','Churn'],axis=1)
# convert target 
y=df['Churn'].apply(lambda x:1 if x =="Yes" else 0)

model=RandomForestClassifier(n_estimators=100,random_state=42)

# train the model 
model.fit(X,y)

# save model 
joblib.dump(model,"model/churn_model.pkl")

print("Model Train Successfully!")
print("Model Saved at: model/chun_model.pkl")