import pandas as pd
import nltk 
from .. import config 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

def phrase_tokenizer(text):

  # Tokenize the job description
  doc = config.nlp(text)

  # Extract noun phrases while retaining the original text
  noun_phrases = []
  for chunk in doc.noun_chunks:
      noun_phrases.append(chunk.text)

  # Tokenized words
  tokens = [token.text for token in doc]
  return tokens

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

def skills_search(job_descriptions):

  # Tokenize the job description
  doc = config.nlp(job_descriptions)

  # Initialize lists to store found skills
  exact_matches = []
  semantic_matches = []

  # Iterate through tokens in the job description
  for token in doc:
      for skill in config.skills:
          # Exact match
          if token.text.lower() == skill.lower():
              exact_matches.append(skill)
          # Semantic similarity
          elif config.nlp(token.text).similarity(config.nlp(skill)) > 0.7: #Edit the threshold here 0.7
              semantic_matches.append(skill)

  # Remove duplicates
  exact_matches = list(set(exact_matches))
  semantic_matches = list(set(semantic_matches))
  return exact_matches, semantic_matches

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