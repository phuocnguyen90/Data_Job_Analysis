import pandas as pd
from config import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight


vectorizer_title = TfidfVectorizer(
        max_features=20,
        stop_words='english',
        lowercase=True,
        token_pattern=r'\w{3,}'
    )

vectorizer_desc = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    lowercase=True,
    token_pattern=r'\w{3,}'
)

def train_neural_network(dataframe, job_to_list=job_categories, save_model=False):
    X_title = vectorizer_title.fit_transform(dataframe['Job_Title']).toarray()
    X_desc = vectorizer_desc.fit_transform(dataframe['Job_Description']).toarray()
    X = np.concatenate((X_title, X_desc), axis=1)
    y = dataframe.loc[:,job_to_list]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))  # Input layer with the number of input features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(6, activation='sigmoid'))  # Output layer with 6 units

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    print(model.summary())
    if save_model == True:
        model.save('trained_model/trained_basic_model.keras')
    return model