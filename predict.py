from config import *

from model_train import Tokenizer, pad_sequences

from tensorflow.keras.models import load_model

def predict_using_saved_model(data_to_predict, model_path):
    loaded_model = load_model(model_path)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data_to_predict)
    sequences = tokenizer.texts_to_sequences(data_to_predict)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    predictions = loaded_model.predict(padded_sequences)
    return predictions

from model_train import train_neural_network 

train_neural_network(dataframe=df,save_weights_path="trained_basic_model.h5" )