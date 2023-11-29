import config
import numpy as np
from functions.prediction import predict_text_cnn
from tensorflow import keras
from keras.models import load_model
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# LOAD THE WHOLE MODEL
# trained_model = load_model('trained_model/trained_textCNN_model.keras')
# updated_df=predict_text_cnn(trained_model, df)


# LOAD THE TRAINED MODEL FROM SAVED WEIGHTS
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

# predict a single job with an array ['Job description','Job title']
# job_to_predict= ['Fundamental knowledge of key machine learning and data science concepts across a number of disciplines such as Natural Language Processing, Social Network Analysis, Time Series Analysis, Computer Vision and others','Data Analyst']

#Create a blank model that matches the architecture of the TextCNN model
blank_model = Sequential()
blank_model.add(Embedding(input_dim=2000, output_dim=100, input_length=120))
blank_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
blank_model.add(MaxPooling1D(pool_size=2))
blank_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
blank_model.add(MaxPooling1D(pool_size=2))
blank_model.add(Flatten())
blank_model.add(Dense(128, activation='relu'))
blank_model.add(Dense(6, activation='sigmoid')) 

# Load the saved weights into this blank model
blank_model.load_weights('trained_model/trained_textCNN_weights.h5')

# Get predictions
# predictions = predict_text_cnn(blank_model, job_to_predict)
# print(predictions)
# Tokenize and pad sequences for 'Job_Description'
category_names = ['Data Engineer', 'Data Analyst', 'Data Scientist', 'Business Analyst', 'Business Intelligence', 'Others']

def format_prediction(predicted_results):
    # Get the first (and only) row from the DataFrame
    row = predicted_results.iloc[0]

    # Extract the probabilities for each category
    probabilities = row.values  # Assuming the DataFrame has the probabilities as values

    # Format the output in a more readable format
    formatted_output = {}
    for category, prob in zip(predicted_results.columns, probabilities):
        formatted_output[category] = f"{prob * 100:.2f}%"

    return formatted_output



def get_category(job_description, job_title, yoe, est_salary):
    # Preprocess the input data
    input_data = [job_description, job_title, yoe, est_salary]
    processed_data = config.preprocess_array(input_data)

    # Get predictions from the model
    result= predict_text_cnn(blank_model,processed_data)

    return result

