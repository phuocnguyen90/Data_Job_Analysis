import pandas as pd
import numpy as np
import nltk 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from spacy.matcher import PhraseMatcher
from nltk.stem import PorterStemmer
import pickle
import csv


# import the config file
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)

import config  # Import config from the parent directory

job_description= "Strong skills in data requirement analysis, analytical thinking, and fast learning; Experience in building dashboards and working with Tableau BI or Power BI is a plus"

skills = config.skills
nlp=config.nlp

df = pd.read_csv('skills.csv')
word_phrase_list = []
with open('skills.csv', newline='', encoding='utf-8') as csvfile:  
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        word_phrase_list.append(row[0]) 
# Create a set to store unique tokens


# Initialize WordNet Lemmatizer
lemmatizer = WordNetLemmatizer()

tokenized_words_phrases = [word_tokenize(word) for word in word_phrase_list]

# Initialize Porter Stemmer
stemmer = PorterStemmer()

# Stem the tokenized words
stemmed_words_phrases = []
for tokenized_phrase in tokenized_words_phrases:
    stemmed_phrase = [stemmer.stem(word) for word in tokenized_phrase]
    stemmed_words_phrases.append(stemmed_phrase)

# Write the stemmed words/phrases to a CSV file
csv_file = "c.csv"

with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(stemmed_words_phrases)
