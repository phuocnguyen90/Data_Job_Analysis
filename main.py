import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import spacy
nltk.download('stopwords')
nltk.download('words')

# importing the dataset
nlp = spacy.load('en_core_web_md')

df = pd.read_csv('data_job_new.csv')

df.head()
print(df.head())
