import pandas as pd
import numpy as np
from .. import config
from salary_predict_function import phrase_tokenizer, skills_search, feature_engineer, feature_select
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

df = config.df

# Pre-generated skills set table. Can be generated with visualization.generate_soft_skill but will take a while
skill_table=pd.read_csv('skill_table.csv')

xgb_df = feature_engineer(df,col_to_lbl=['Location'])
xgb_df = config.preprocess_dataframe(xgb_df,remove_outliers=True)

features = ['YOE', 'Min_level', 'YOE', 'Location', 'Overseas', 'VN']
categorical_columns = ['Location']

X_train, X_test, y_train, y_test = feature_select(xgb_df, features, skill_table)

print('Dataset preparation complete.')
print(f'Number of features: {len(X_train)}')
print(f'Dataset length: {len(xgb_df)}')

#XGB
#salary_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) 
#salary_xgb_model.fit(X_train, y_train)

# Make predictions
#y_pred = salary_xgb_model.predict(X_test)

# Evaluate the model
#mae = mean_absolute_error(y_test, y_pred)

#print("Mean Absolute Error:", mae)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create the random forest regressor
rf = RandomForestRegressor(random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_rf = RandomForestRegressor(random_state=42, **best_params)
best_rf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_rf.predict(X_test)

# Best Hyperparameters: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 20}

mae = mean_absolute_error(y_test, y_pred)

print("Mean Absolute Error:", mae)