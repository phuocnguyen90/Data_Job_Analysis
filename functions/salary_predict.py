import pandas as pd
from salary_predict_function import *
from sklearn.metrics import mean_absolute_error
import pickle

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)
import config  

df = config.preprocess_data(config.test)

df = df.dropna(subset=['Est_Salary'])

def train_and_save_model(df):
    # Load the data
    X_train, X_test, y_train, y_test = load_data(df)
    # Train the model
    trained_model = train_model(X_train, X_test, y_train, y_test)
    # Run predictions using the trained model
    predictions, mae = run_prediction(trained_model, X_test, y_test)

    # Check if the trained model's MAE is smaller than the stored model
    save_model = True
    if os.path.exists('trained_model/salary_model.pkl'):
        with open('trained_model/salary_model.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        saved_mae = loaded_data['mae']
        if mae >= saved_mae:
            user_input = input("The new model's performance is not better. Do you want to overwrite the existing model? (y/n): ")
            if user_input.lower() != 'y':
                save_model = False  # Don't save the new model if the user declines

    # Save the model only if the new MAE is smaller or if the user agrees to overwrite
    if save_model:
        model_and_metrics = {
            'model': trained_model,
            'mae': mae
        }

        with open('trained_model/salary_model.pkl', 'wb') as file:
            pickle.dump(model_and_metrics, file)

train_and_save_model(df)

def salary_predict(df): #  expected columns = ['YOE', 'Min_level', 'Location', 'Overseas', 'VN']
    
    X_train, X_test, y_train, y_test = load_data(df)
    # Load the saved model
    with open('trained_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
    predictions = loaded_model.predict(X_test)

    return predictions

job_pred=['2','0','0','0','0']