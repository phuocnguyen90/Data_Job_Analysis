import pandas as pd
import numpy as np
from config import *
from sklearn.model_selection import train_test_split
#placeholder skills set. Can be generated with visualization.generate_soft_skill

skills=['A/B Testing', 'Adaptability', 'Agile Methodologies', 'AI Ethics', 'AI Explainability', 'AI Fairness',
        'API Integration', 'Attention to Detail', 'Big Data', 'Business Acumen', 'Business Analysis',
        'Business Intelligence', 'Change Management', 'Classification', 'Client Management', 'Cloud Computing',
        'Clustering', 'Collaboration', 'Communication', 'Conflict Resolution', 'Containers', 'Creative Problem-Solving',
        'Creative Thinking', 'Critical Thinking', 'Dashboard Creation', 'Dashboard', 'Data Analysis',
        'Data Analytics', 'Data Architecture', 'Data Auditing', 'Data Augmentation', 'Data Catalog', 'Data Cleaning',
        'Data Cleansing', 'Data Collection', 'Data Driven', 'Data Engineering', 'Data Ethics', 'Data Exploration',
        'Data Governance', 'Data Imputation', 'Data Integration', 'Data Interpretation', 'Data Lake',
        'Data Lake Management', 'Data Lineage', 'Data Management', 'Data Manipulation', 'Data Migration', 'Data Mining',
        'Data Modeling', 'Data Monetization', 'Data Normalization', 'Data Pipelines', 'Data Preprocessing', 'Data Privacy',
        'Data Profiling', 'Data Quality', 'Data Reporting', 'Data Reshaping', 'Data Science',
        'Data Scrubbing', 'Data Security', 'Data Stewardship', 'Data Storytelling', 'Data Strategy', 'Data Testing',
        'Data Transformation', 'Data Validation', 'Data Visualization', 'Data Warehousing',
        'Data Wrangling', 'Database Design', 'Database Management', 'Data-Driven',
        'Decision-Making', 'Deep Learning', 'Dimensionality Reduction', 'Distributed Computing', 'Django', 'Docker',
        'Domain Knowledge', 'EDA', 'Ensemble', 'Ethical', 'ETL', 'Excel', 'Experimental Design',
        'FastAPI', 'Feature Engineering', 'Feature Scaling', 'Feature Selection', 'flask', 'Frameworks', 'GDPR',
        'Geospatial Analysis', 'Git', 'Gradient Boosting', 'Hadoop', 'Hypothesis Testing', 'Informatica', 'Interpersonal Skills',
        'Kanban', 'Leadership', 'Libraries', 'Machine Learning', 'Mathematics', 'Matplotlib', 'ML', 'Model Deployment',
        'Model Interpretation', 'Natural Language Processing ', 'NLP', 'nltk', 'NoSQL', 'Numpy', 'Pandas', 'Power BI',
        'Predictive Modeling', 'Presentation', 'Privacy compliance', 'Problem Definition', 'Problem Solving',
        'Project Management', 'Project Planning', 'Python', 'PyTorch', 'Quantitative Analysis', 'Query Optimization',
        'R', 'Random Forest', 'Regression Analysis', 'Reinforcement Learning', 'Requirements Gathering',
        'Sampling Techniques', 'Scikit-learn', 'Scrum', 'spacy', 'Spark', 'SQL', 'Stakeholder Management', 'Statistical Analysis',
        'Statistical Testing', 'Statistical Thinking', 'Strategic Thinking', 'Supervised Learning', 'Tableau', 'Teamwork',
        'TensorFlow', 'Text Processing', 'Time Management', 'Time Series',
        'Unsupervised Learning','vba', 'Version Control ', 'Web Scraping', 'Seaborn', 'XGBoost',
]


def phrase_tokenizer(text):

  # Tokenize the job description
  doc = nlp(text)

  # Extract noun phrases while retaining the original text
  noun_phrases = []
  for chunk in doc.noun_chunks:
      noun_phrases.append(chunk.text)

  # Tokenized words
  tokens = [token.text for token in doc]
  return tokens

def skills_search(job_descriptions):

  # Tokenize the job description
  doc = nlp(job_descriptions)

  # Initialize lists to store found skills
  exact_matches = []
  semantic_matches = []

  # Iterate through tokens in the job description
  for token in doc:
      for skill in skills:
          # Exact match
          if token.text.lower() == skill.lower():
              exact_matches.append(skill)
          # Semantic similarity
          elif nlp(token.text).similarity(nlp(skill)) > 0.7: #Edit the threshold here 0.7
              semantic_matches.append(skill)

  # Remove duplicates
  exact_matches = list(set(exact_matches))
  semantic_matches = list(set(semantic_matches))
  return exact_matches, semantic_matches

def create_skill_encoding_table(skills, job_descriptions):
    # Create an empty DataFrame with columns for each skill
    skill_encoding_table = pd.DataFrame(columns=skills, dtype=bool)

    # Loop through job descriptions and search for skills
    for job_description in job_descriptions:
        tokens = phrase_tokenizer(job_description)

        # Create a dictionary to represent skill presence (1 if found, 0 otherwise)
        skill_presence = {skill: 1 if skill.lower() in [token.lower() for token in tokens] else 0 for skill in skills}
        # Create a DataFrame from the skill_presence dictionary
        skill_presence_df = pd.DataFrame([skill_presence])

        # Concatenate the skill_presence_df to the skill_encoding_table
        skill_encoding_table = pd.concat([skill_encoding_table, skill_presence_df], ignore_index=True)

    return skill_encoding_table

# Create the skill encoding table

skill_table = create_skill_encoding_table(skills, df['Job_Description'])

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Format the features
def feature_engineer(df, col_to_avg=None, col_to_encode=None, col_to_lbl=None, fill='NaN'):

    # Average the cols
    if col_to_avg is not None and len(col_to_avg) > 1:
        df['col_avg'] = df[col_to_avg].mean(axis=1)

    # One-hot encode the cols
    if col_to_encode is not None:
        df = pd.get_dummies(df, columns=col_to_encode, drop_first=True)

    # Label encode the cols
    if col_to_lbl is not None:
        label_encoder = LabelEncoder()
        for col in col_to_lbl:
            df[col] = label_encoder.fit_transform(df[col])

    # Fill the missing values
    if fill == 0:
        df.fillna(value=float(0), inplace=True)
    elif fill == 'NaN':
        df.fillna(value=float('NaN'), inplace=True)

    return df

# Select the feature
def feature_select(df, features, skills=None):
    if skills is None:
        features = features
        target = 'Est_Salary'
    elif skills is not None:
        features = features+list(skills.columns)
        target = 'Est_Salary'
        df = df.join(skills, how='left')

    X = df[features]
    y = df[target]
    # Replace NaN values with 0 in your feature matrix (X)
    X = X.fillna(0)
    # Replace NaN values with 0 in your target variable (y) if needed
    y = y.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

xgb_df = feature_engineer(df,col_to_lbl=['Location'])

features = ['YOE', 'Min_level', 'YOE', 'Location', 'Overseas', 'VN']
categorical_columns = ['Location']

X_train, X_test, y_train, y_test = feature_select(xgb_df, features, skill_table)

#XGB
#salary_xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) 
#salary_xgb_model.fit(X_train, y_train)

# Make predictions
#y_pred = salary_xgb_model.predict(X_test)

# Evaluate the model
#mae = mean_absolute_error(y_test, y_pred)

#print("Mean Absolute Error:", mae)

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create the random forest regressor
rf = RandomForestRegressor(random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = random_search.best_params_
print("Best Hyperparameters:", best_params)

# Train the model with the best hyperparameters
best_rf = RandomForestRegressor(random_state=42, **best_params)
best_rf.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = best_rf.predict(X_test)

# Best Hyperparameters: {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 4, 'max_features': 'auto', 'max_depth': 20}

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)