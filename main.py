# Import 
import pandas as pd
import numpy as np
import nltk
from nltk import stopwords
from textblob import Word

def preprocess_job_description(dataframe, remove_outliers=False):
    # Function for preprocessing job descriptions
    dataframe['Job_Description'] = dataframe['Job_Description'].astype(str)
    dataframe['Job_Title'] = dataframe['Job_Title'].astype(str)
    dataframe = dataframe.dropna(subset=['Job_Description'])

    # Lower case
    dataframe['Job_Description'] = dataframe['Job_Description'].apply(lambda x: " ".join(x.lower() for x in x.split()))

    # Remove tabulation and punctuation
    dataframe['Job_Description'] = dataframe['Job_Description'].str.replace('[^\w\s]',' ')

    # Remove digits
    dataframe['Job_Description'] = dataframe['Job_Description'].str.replace('\d+', '')

    # Remove stop words
    stop = stopwords.words('english')
    dataframe['Job_Description'] = dataframe['Job_Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # Lemmatization
    dataframe['Job_Description'] = dataframe['Job_Description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

    if remove_outliers==True:

        # Limit years of experience to a maximum of 10

        dataframe['YOE'] = dataframe['YOE'].apply(lambda x: min(x, 10))
        # Remove the outlier jobs that have Est_Salary larger than $10,000
        dataframe['Est_Salary'] = dataframe['Est_Salary'].apply(lambda x: min(x, 10000))


    return dataframe  # Return the processed DataFrame


