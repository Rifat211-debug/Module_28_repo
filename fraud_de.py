#============================
#Import Required Libraries
#============================
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix

#===================
#Load Dataset
#===================

df = pd.read_csv('Health Insurance Fraud Claims.csv')

#============================
#Drop the Unuseful columns
#============================

df.drop(columns = ['ProviderID','ClaimID', 'PatientID', 'ClaimDate', 'DiagnosisCode', 'ProcedureCode', 'Cluster'], inplace = True)

#====================
#Level  encoding
#====================

df['ClaimLegitimacy'] = df['ClaimLegitimacy'].map({'Legitimate' : 0, 'Fraud' : 1})

#====================
#Terget & Features
#===================

X = df.drop(columns = ['ClaimLegitimacy'])
y = df['ClaimLegitimacy']

#================
#Column Split
#================

num_col = X.select_dtypes(include = ['int64', 'float64']).columns
cat_col = X.select_dtypes(include = ['object']).columns

#=================
#Preprocessing
#=================

num_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'median')),
        ('scaler', StandardScaler())
    ]
)

cat_transformer = Pipeline(
    steps = [
        ('imputer', SimpleImputer(strategy = 'most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_col),
        ('cat', cat_transformer, cat_col)
    ]
)

#========================
#Gradient Boosting Model
#========================

gb_model = GradientBoostingClassifier(
    n_estimators = 50,
    max_depth = None,
    min_samples_split = 2,
    learning_rate = 0.1,
    random_state = 42
)

#=================
#Full Pipeline
#=================

gb_pipeline = Pipeline(steps = [
    ('preprocess', preprocessor),
    ('model', gb_model)
])

#===================
#Train Test Split
#===================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.25,
    random_state = 42
)

#============================
#Train & Predict
#============================

gb_pipeline.fit(X_train, y_train)
y_pred = gb_pipeline.predict(X_test)

#=====================
#Evaluation
#=====================

rec = recall_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"Recall : {rec:.4f}")
print(f"Accuracy : {acc:.4f}")

#==================
#Save Model
#==================

with open('fraud_detection_pipeline.pkl', 'wb') as file:
    pickle.dump(gb_pipeline, file)

print("===> Gradient Boosting Pipeline Saved as fraud_detection_pipeline.pkl <===")  