import pandas as pd
import numpy as np

import tensorflow as tf
import kerastuner as kt 
from tensorflow import keras
from keras import layers
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)

import config  # Import config from the parent directory


def train_text_cnn(dataframe, job_categories=config.job_categories,save_model=False):
    # Define X as the job descriptions (JD_Trans)
    X_desc = dataframe['Job_Description']
    X_title = dataframe['Job_Title']

    # Define y as the one-hot encoded job categories
    y = dataframe[job_categories]
    
    # Split the data into training and testing sets for 'Job_Description'
    X_train_desc, X_test_desc, X_train_title, X_test_title, y_train, y_test = train_test_split(X_desc, X_title, y, test_size=0.2, random_state=109)

    # Tokenize and pad sequences for 'Job_Description'
    tokenizer_desc = Tokenizer(num_words=2000)
    tokenizer_desc.fit_on_texts(X_train_desc)
    X_train_desc = tokenizer_desc.texts_to_sequences(X_train_desc)
    X_test_desc = tokenizer_desc.texts_to_sequences(X_test_desc)
    X_train_desc = pad_sequences(X_train_desc, maxlen=100)
    X_test_desc = pad_sequences(X_test_desc, maxlen=100)

    # Tokenize and pad sequences for 'Job_Title'
    tokenizer_title = Tokenizer(num_words=100)
    tokenizer_title.fit_on_texts(X_train_title)
    X_train_title = tokenizer_title.texts_to_sequences(X_train_title)
    X_test_title = tokenizer_title.texts_to_sequences(X_test_title)
    X_train_title = pad_sequences(X_train_title, maxlen=20)
    X_test_title = pad_sequences(X_test_title, maxlen=20)

    # Combine 'Job_Description' and 'Job_Title' sequences
    X_train = np.concatenate((X_train_desc, X_train_title), axis=1)
    X_test = np.concatenate((X_test_desc, X_test_title), axis=1)

    num_labels = 6
    CNN_model = Sequential()
    CNN_model.add(Embedding(input_dim=2000, output_dim=100, input_length=120))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(128, activation='relu'))
    CNN_model.add(Dense(num_labels, activation='sigmoid'))
    CNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    CNN_model.fit(X_train, y_train, epochs=8, batch_size=32, validation_data=(X_test, y_test))
    
    loss, accuracy = CNN_model.evaluate(X_test, y_test)  

    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')
    print(CNN_model.summary())
    
    if save_model:
        CNN_model.save_weights('trained_model/trained_textCNN_weights.h5') # we can save() or save_weights
    
    return CNN_model




