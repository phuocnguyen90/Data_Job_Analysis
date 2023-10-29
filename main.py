# Import 
import pandas as pd
import numpy as np



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN
from sklearn.preprocessing import MultiLabelBinarizer


import nltk
from nltk import stopwords
from textblob import Word

# Function Definitions

def preprocess_job_description(test, remove_outliers=False):
    # Function for preprocessing job descriptions
    test['Job_Description'] = test['Job_Description'].astype(str)
    test['Job_Title'] = test['Job_Title'].astype(str)

    # Lower case
    test['Job_Description'] = test['Job_Description'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove tabulation and punctuation
    test['Job_Description'] = test['Job_Description'].str.replace('[^\w\s]',' ')

    # Remove digits
    test['Job_Description'] = test['Job_Description'].str.replace('\d+', '')

    # Remove stop words
    stop = stopwords.words('english')
    test['Job_Description'] = test['Job_Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Lemmatization
    test['Job_Description'] = test['Job_Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    if remove_outliers==True:

        # Limit years of experience to a maximum of 10

        test['YOE'] = test['YOE'].apply(lambda x: min(x, 10))
        # Remove the outlier jobs that have Est_Salary larger than $10,000
        test['Est_Salary'] = test['Est_Salary'].apply(lambda x: min(x, 10000))


    return test  # Return the processed DataFrame


