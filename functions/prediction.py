import pandas as pd
import numpy as np
import tensorflow as tf
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


def predict_text_cnn(model, data, job_categories=config.job_categories):
    if isinstance(data, pd.DataFrame):
        X_desc = data['Job_Description']
        X_title = data['Job_Title']
    elif isinstance(data, list) and len(data) == 2:
        X_desc = [data[0]]
        X_title = [data[1]]
    else:
        raise ValueError("Input data format not recognized. Please provide a DataFrame or a list of two elements.")

    tokenizer_desc = Tokenizer(num_words=2000)
    tokenizer_desc.fit_on_texts(X_desc)
    X_desc = tokenizer_desc.texts_to_sequences(X_desc)
    X_desc = pad_sequences(X_desc, maxlen=100)

    tokenizer_title = Tokenizer(num_words=100)
    tokenizer_title.fit_on_texts(X_title)
    X_title = tokenizer_title.texts_to_sequences(X_title)
    X_title = pad_sequences(X_title, maxlen=20)

    X = np.concatenate((X_desc, X_title), axis=1)

    predictions = model.predict(X)
    y_test_predicted = pd.DataFrame(predictions, columns=job_categories)
    
    return y_test_predicted

#def predict_salary(model,data, job_categories=config.job_categories):

#    predictions = model.predict(X)
#    y_test_predicted = pd.DataFrame(predictions, columns=job_categories)
