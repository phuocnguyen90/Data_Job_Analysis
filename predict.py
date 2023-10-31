from config import *
from functions.prediction import predict_text_cnn
from tensorflow.keras.models import load_model

# LOAD THE WHOLE MODEL
# trained_model = load_model('trained_model/trained_textCNN_model.keras')
# updated_df=predict_text_cnn(trained_model, df)


# LOAD THE TRAINED MODEL FROM SAVED WEIGHTS
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense

# predict a single job with an array ['Job description','Job title']
job_to_predict= ['Fundamental knowledge of key machine learning and data science concepts across a number of disciplines such as Natural Language Processing, Social Network Analysis, Time Series Analysis, Computer Vision and others','Data Analyst']

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

def predict_salary(job_description, job_title):
    # Preprocess the input data
    processed_data = preprocess_data(job_description, job_title)

    # Get predictions from the model
    predicted_salary = blank_model.predict(processed_data)

    return predicted_salary