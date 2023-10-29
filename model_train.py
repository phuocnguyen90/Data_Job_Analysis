from config import *

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from classifier import vectorizer



def train_neural_network(dataframe, save_weights_path, job_categories=job_categories ):
    X = vectorizer.fit_transform(dataframe['Job_Description']).toarray()
    y = dataframe[job_categories]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))  # Input layer with the number of input features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))  # Output layer with 5 units

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    model.save('trained_basic_model.h5')




def train_and_save_text_cnn(dataframe, save_weights_path, job_categories=job_categories):
    # Define X as the job descriptions (JD_Trans)
    X = dataframe['Job_Description']
    # Define y as the one-hot encoded job categories
    y = dataframe[job_categories]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)

    train_texts = X_train
    test_texts = X_test

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)

    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)

    num_labels = 5
    CNN_model = Sequential()
    CNN_model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(128, activation='relu'))
    CNN_model.add(Dense(num_labels, activation='sigmoid'))

    CNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    CNN_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model's weights
    CNN_model.save('trained_model.h5')