# global variables

import pandas as pd
import numpy as np
import spacy
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


job_categories = ['Data_Engineer', 'Data_Analyst', 'Data_Scientist', 'Business_Analyst', 'Business_Intelligence', 'Others']

# A list of technical skills that are needed in general
technical_skills = [
    'python','r','c','c++','java','hadoop','scala','flask','pandas',
    'spark','scikit','numpy','php','sql','mysql','css','mongodb','nltk',
    'keras','pytorch','tensorflow','linux','ruby','javascript','django',
    'react','reactjs','ai','artificial intelligence','ui', 'skicit',
    'tableau','power bi','machine learning','frontend','big data',
    'data mining','data warehousing','data visualization','data engineering',
    'data modeling','data governance','data analytics','statistical analysis',
    'natural language processing', 'computer vision','deep learning',
    'data preprocessing','etl','data quality management','excel','vba','gcp']
# A list of soft skills that are needed in general
combined_soft_skills = [
    "Communication", "Critical Thinking", "Creativity","Adaptability", "Teamwork", "Attention to Detail",
    "Time Management", "Emotional Intelligence", "Empathy","Conflict Resolution", "Decision-Making", "Leadership",
    "Problem-Solving", "Ethical Judgment", "Flexibility","Customer Service", "Negotiation", "Innovation",
    "Persuasion", "Resilience", "Collaboration", "Networking","Conflict Management", "Cultural Sensitivity", "Stress Management",
    "Self-Motivation", "Open-Mindedness", "Information Presentation","Risk Management", "Active Listening"
]
# meaningless words that should be manually removed
remove_words = ['data','year','least','working','business',
        'tool','position','related','strong','field',
        'minimum','good','salary','work','using','industry',
        'processing','hn','kinh','nghi','experience','relevant'
    ]

file_path_train = os.path.join('dataset', 'Data_Jobs.csv')
file_path_deploy = os.path.join('dataset', 'data_job_new.csv')


# Importing the dataset
nlp = spacy.load('en_core_web_md')
df = pd.read_csv(file_path_deploy)
test = pd.read_csv(file_path_train)


from textblob import Word

def preprocess_array(input_data):
    # Create a DataFrame with the provided input elements
    dataframe = pd.DataFrame({
        'Job_Description': [input_data[0]],
        'Job_Title': [input_data[1]],
        'YOE': [input_data[2]],
        'Est_Salary': [input_data[3]]
    })

    return preprocess_data(dataframe)

def preprocess_data(dataframe, remove_outliers=False):
    dataframe = dataframe.copy()  # Create a copy of the DataFrame

    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input data should be a DataFrame.")

    # Convert columns to string where needed
    for column in ['Job_Description', 'Job_Title']:
        if dataframe[column].dtype != 'object':
            dataframe[column] = dataframe[column].astype(str)

    # Handle missing values in 'Job_Description'
    dataframe = dataframe.dropna(subset=['Job_Description'])

    # Text preprocessing
    dataframe['Job_Description'] = dataframe['Job_Description'].str.lower()
    dataframe['Job_Description'] = dataframe['Job_Description'].str.replace('[^\w\s]',' ')
    dataframe['Job_Description'] = dataframe['Job_Description'].str.replace('\d+','')
    stop = set(stopwords.words('english'))
    dataframe['Job_Description'] = dataframe['Job_Description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    if remove_outliers:
        # Limit years of experience to a maximum of 10
        dataframe['YOE'] = dataframe['YOE'].apply(lambda x: min(x, 10))
        # Remove the outlier jobs that have Est_Salary larger than $10,000
        dataframe['Est_Salary'] = dataframe['Est_Salary'].apply(lambda x: min(x, 10000))

    return dataframe