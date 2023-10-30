import numpy as np
from config import *
import main
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, concatenate
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data=main.preprocess_job_description(test)


# Preprocessing
max_words = 1000  # Maximum number of words to tokenize
max_sequence_length = 300  # Maximum sequence length
num_categories = 6  # Number of categories

tokenizer_title = Tokenizer(num_words=max_words)
tokenizer_title.fit_on_texts(data['Job_Title'])
sequences_title = tokenizer_title.texts_to_sequences(data['Job_Title'])
data_title = pad_sequences(sequences_title, maxlen=max_sequence_length)

tokenizer_desc = Tokenizer(num_words=max_words)
tokenizer_desc.fit_on_texts(data['Job_Description'])
sequences_desc = tokenizer_desc.texts_to_sequences(data['Job_Description'])
data_desc = pad_sequences(sequences_desc, maxlen=max_sequence_length)



input_title = Input(shape=(max_sequence_length,))
input_desc = Input(shape=(max_sequence_length,))

embedding_title = Embedding(max_words, 100, input_length=max_sequence_length)(input_title)
embedding_desc = Embedding(max_words, 100, input_length=max_sequence_length)(input_desc)

conv1d_title = Conv1D(128, 5, activation='relu')(embedding_title)
conv1d_desc = Conv1D(128, 5, activation='relu')(embedding_desc)

global_max_pooling_title = GlobalMaxPooling1D()(conv1d_title)
global_max_pooling_desc = GlobalMaxPooling1D()(conv1d_desc)

concatenated = concatenate([global_max_pooling_title, global_max_pooling_desc])

X_desc = data_desc
X_title = data_title

# Define y as the one-hot encoded job categories
y = test[job_categories]

# Split the data into training and testing sets for 'Job_Description'
X_train_desc, X_test_desc, X_train_title, X_test_title, y_train, y_test = train_test_split(X_desc, X_title, y, test_size=0.2, random_state=109)



dense = Dense(64, activation='relu')(concatenated)
dropout = Dropout(0.2)(dense)
output = Dense(num_categories, activation='sigmoid')(dropout)

model = Model(inputs=[input_title, input_desc], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
model.fit([X_train_title, X_train_desc], y_train, epochs=10, batch_size=32, validation_split=0.3)

# Evaluate the model
loss, accuracy = model.evaluate([X_test_title, X_test_desc], y_test)
print(f'Accuracy: {accuracy}')

# For inference/predictions
predictions = model.predict([X_test_title, X_test_desc])
# Use these predictions to evaluate the model's performance further
