#============================
#Import Required Libraries
#============================

import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#===================
#Load Dataset
#===================

df = pd.read_csv('insurance.csv')
print(df)

#============================
#Drop the Duplicates Rows
#============================

df = df.drop_duplicates(keep = 'first')

#====================
#Terget & Features
#===================

X = df.drop(columns = ['charges'])
y = df['charges']

#================
#Column Split
#================

num_col = X.select_dtypes(include = ['int64', 'float64']).columns
cat_col = X.select_dtypes(include = ['object']).columns

#==========================
#Preprocessing Pipeline
#==========================

num_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ('num', num_transformer, num_col),
        ('cat', cat_transformer, cat_col)
    ]
)

#========================
#Gradient Boosting Model
#========================

gb_model = GradientBoostingRegressor(
    n_estimators = 300,
    learning_rate = 0.01,
    max_depth = 5,
    min_samples_split = 2,
    random_state = 42
)

#=================
#Full Pipeline
#=================

gb_pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('model', gb_model)
])

#===================
#Train Test Split
#===================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.20,
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

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.4f}")
print(f"R2 : {r2:.4f}")

#==================
#Save Model
#==================

with open('medical_cost_prediction.pkl', 'wb') as file:
    pickle.dump(gb_pipeline, file)

print("===> Gradient Boosting Pipeline Saved as medical_cost_prediction.pkl <===")      
