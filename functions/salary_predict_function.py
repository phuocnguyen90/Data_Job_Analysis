import pandas as pd
import numpy as np
import nltk 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle

# import the config file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)

import config  # Import config from the parent directory

label_encoder = LabelEncoder()

# Create the skill encoding table
def skills_search(job_data):
    if isinstance(job_data, pd.DataFrame) and 'Job_Description' in job_data.columns:
        # If input is a DataFrame with 'Job_Description' column
        job_descriptions = job_data['Job_Description'].tolist()

        # Initialize a DataFrame to store matches
        matches_df = pd.DataFrame(columns=['Job_Description'] + config.skills)

        for description in job_descriptions:
            # Initialize a row with all False values
            row = {col: False for col in matches_df.columns}
            row['Job_Description'] = description

            doc = config.nlp(description)
            for token in doc:
                for skill in config.skills:
                    # Exact match
                    if token.text.lower() == skill.lower():
                        row[skill] = True
                        break
                    # Semantic similarity
                    elif token.text and skill and config.nlp(token.text).has_vector and config.nlp(skill).has_vector:
                        similarity_score = config.nlp(token.text).similarity(config.nlp(skill))
                        if similarity_score > 0.7:
                            row[skill] = True


            
            matches_df = pd.concat([matches_df, pd.DataFrame([row])], ignore_index=True)

        return matches_df

    elif isinstance(job_data, (list, pd.Series)):
        # If input is an array or list of job descriptions
        # Create a DataFrame with 'Job_Description' and predefined skills as columns
        matches_df = pd.DataFrame(columns=['Job_Description'] + config.skills)

        for description in job_data:
            # Initialize a row with all False values
            row = {col: False for col in matches_df.columns}
            row['Job_Description'] = description

            doc = config.nlp(description)
            for token in doc:
                for skill in config.skills:
                    # Exact match
                    if token.text.lower() == skill.lower():
                        row[skill] = True
                    # Semantic similarity
                    elif config.nlp(token.text).similarity(config.nlp(skill)) > 0.7:
                        row[skill] = True

            matches_df = pd.concat([matches_df, pd.DataFrame([row])], ignore_index=True)

        return matches_df

    else:
        raise ValueError("Invalid input format. Please provide a DataFrame with 'Job_Description' column or an array/list of job descriptions.")

# Format the features
def feature_engineer(df, col_to_avg=None, col_to_encode=None, col_to_lbl=None, fill='NaN'):

    # Average the cols
    if col_to_avg is not None and len(col_to_avg) > 1:
        df['col_avg'] = df[col_to_avg].mean(axis=1)

    # One-hot encode the cols
    if col_to_encode is not None:
        df = pd.get_dummies(df, columns=col_to_encode, drop_first=True)

    # Label encode the cols
    if col_to_lbl is not None:
        label_encoder = LabelEncoder()
        for col in col_to_lbl:
            df[col] = label_encoder.fit_transform(df[col])

    # Fill the missing values
    if fill == 0:
        df.fillna(value=float(0), inplace=True)
    elif fill == 'NaN':
        df.fillna(value=float('NaN'), inplace=True)

    return df

# Select the feature
def feature_select(df, features, skills=None):
    if skills is None:
        features = features
        
    elif skills is not None:
        features = features+list(skills.columns)
        
        df = df.join(skills, how='left')
    
    X = df[features]
    y = df['Est_Salary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_data(df):
    # Pre-generated skills set table. Can be generated with visualization.generate_soft_skill but will take a while
    skill_table = skills_search(df)
    features = ['YOE', 'Min_level', 'Location', 'Overseas', 'VN']
    categorical_columns = []
    col_to_encode = ['Location']

    if isinstance(df, pd.DataFrame):
        # If the input is already a DataFrame
        # Check if all necessary columns exist, if not, fill missing columns with zeros
        for column in features:
            if column not in df.columns:
                df[column] = 0

    else:
        # If the input is a list or an array
        # Convert the list or array to a DataFrame
        if isinstance(df, (list, pd.Series)):
            df = pd.DataFrame(df)
        elif isinstance(df, np.ndarray):
            df = pd.DataFrame(df)

        for column in features:
            if column not in df.columns:
                df[column] = 0


    xgb_df = feature_engineer(df, col_to_encode=col_to_encode)
    xgb_df = feature_select(df,features,skill_table)
    xgb_df = config.preprocess_data(xgb_df)

    X_train, X_test, y_train, y_test = feature_select(xgb_df, features, skill_table)

    return X_train, X_test, y_train, y_test

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

def train_model(X_train, X_test, y_train, y_test):
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

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)

    return best_rf  # Return the trained model

def run_prediction(model, X_test, y_test):
    # Predict using the trained model
    y_pred = model.predict(X_test)
    # Evaluate the model on the test set
    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error:", mae)
    return y_pred, mae # Return predictions and MAE
