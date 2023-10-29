# Import 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string
import keras
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.util import ngrams
from wordcloud import WordCloud
import spacy
from collections import Counter
from textblob import Word
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
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
import xgboost as xgb
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.preprocessing.text import Tokenizer

nltk.download('stopwords')
nltk.download('words')

# importing the dataset
nlp = spacy.load('en_core_web_md')

df = pd.read_csv('data_job_new.csv')
test = pd.read_csv('Data_Jobs.csv')

jda_clean = test.dropna(subset=['Job_Description', 'Data_Engineer', 'Data_Analyst', 'Data_Scientist', 'Business_Analyst', 'Business_Intelligence'])


# global variables
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


def plot_job_category_counts(job_categories, test):
    category_counts = []

    # Count the number of occurrences for each job category
    for category in job_categories:
        count = test[category].sum()
        print(f'Category: {category}, Count: {count}')
        category_counts.append(count)

    # Print the list of counts
    print("Category Counts:", category_counts)
    colors = ['blue', 'green', 'red', 'orange', 'purple']

    # Create a histogram
    plt.figure(figsize=(10, 5))
    plt.bar(job_categories, category_counts, color=colors)
    for i, value in enumerate(category_counts):
        plt.text(i, category_counts[i], str(category_counts[i]), ha='center', va='bottom')
    plt.ylim(0, 1400)
    plt.xlabel('Job Categories')
    plt.ylabel('Count')
    plt.title('Job Count In Each Category')
    plt.xticks(rotation=45)
    plt.show()

def plot_salary_distribution_by_experience(test):

    # Define bin intervals for 'Min_YOE'
    bins = [0, 1, 2, 5, 7, 10, float('inf')]

    # Categorize 'Min_YOE' into bins and create a new column 'Min_YOE_Bin'
    test['YOE_Bin'] = pd.cut(test['YOE'], bins=bins, labels=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)], right=False)

    # Extract data for the box plot
    boxplot_data = [test['Est_Salary'][test['YOE_Bin'] == bin_label].dropna().values for bin_label in test['YOE_Bin'].cat.categories]

    # Create the box plot with custom colors for median (orange) and mean (blue)
    plt.figure(figsize=(10, 15))
    plt.xticks(rotation=45)
    plt.title("Salary Distribution by Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Estimated Salary")

    # Calculate median values for each bin
    median_values = [np.median(data) for data in boxplot_data]

    # Create the box plot with custom colors
    box_plot = plt.boxplot(boxplot_data, labels=test['YOE_Bin'].cat.categories, patch_artist=True)

    # Apply colors to the boxes (using different colors for mean and median)
    box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    median_color = 'orange'

    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)

    # Calculate and add mean value labels to each box
    for i, data in enumerate(boxplot_data):
        mean_salary = int(round(np.mean(data)))
        plt.text(i+1, mean_salary, mean_salary, horizontalalignment='center', verticalalignment='bottom', color='blue')

    # Add median value labels to each box with orange color
    for i, median_val in enumerate(median_values):
        plt.text(i+1, median_val, int(median_val), horizontalalignment='center', verticalalignment='top', color='orange')

    # Create a legend for mean and median colors
    mean_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Mean')
    median_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Median')
    plt.legend(handles=[mean_patch, median_patch])

    plt.show()


def plot_top_locations(test, n=15):
    location_counts = test['Location'].value_counts()

    # To see the top N locations with the most jobs, you can use:
    top_n_locations = location_counts.head(n)

    # Create a bar chart with a logarithmic scale on the y-axis and data labels
    plt.figure(figsize=(10, 6))
    bars = plt.bar(top_n_locations.index, top_n_locations, color='skyblue')

    # Add data labels to each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha='center', va='bottom')

    plt.yscale('log')  # Scale the y-axis by a logarithmic scale
    plt.title("Top Locations with the Most Jobs (Log Scale)")
    plt.xlabel("Location")
    plt.ylabel("Number of Jobs (Log Scale)")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Show the bar chart
    plt.show()


def generate_job_description_word_cloud(test):
    all_descriptions = " ".join(test['Job_Description'])

    all_descriptions = all_descriptions.lower()
    all_descriptions = all_descriptions.translate(str.maketrans('', '', string.punctuation))
    all_descriptions = "".join(filter(lambda x: not x.isdigit(), all_descriptions))

    words = nltk.word_tokenize(all_descriptions)

    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]

    phrases = ["power bi"]

    for i in range(len(filtered_words)):
        if i < len(filtered_words) - 1 and filtered_words[i] + " " + filtered_words[i+1] in phrases:
            filtered_words[i] = filtered_words[i] + " " + filtered_words[i+1]
            filtered_words[i+1] = ""


    filtered_words = [word for word in filtered_words if word not in remove_words]

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud of Most Frequent Words in Job Descriptions')
    plt.show()



def calculate_top_skills_per_category(jda, technical_skills, remove_words):
    top_technical_skills = {}
    top_soft_skills = {}

    categories = job_categories

    for category in categories:
        descriptions = jda[jda[category] == 1]['Job_Description']
        total_jobs = len(descriptions)
        category_words = []

        for desc in descriptions:
            words = desc.lower().split()
            category_words.extend([word for word in words if word not in remove_words])

        skill_counts = Counter(word for word in category_words if word in technical_skills)
        technical = [(skill, count) for skill, count in skill_counts.items() if skill in technical_skills]
        soft = [(skill, count) for skill, count in skill_counts.items() if skill not in technical_skills]

        technical.sort(key=lambda x: x[1], reverse=True)
        soft.sort(key=lambda x: x[1], reverse=True)

        top_technical_skills[category] = [skill for skill, _ in technical[:10]]

        sorted_words = Counter(category_words).most_common()
        top_soft_skills[category] = [word for word, _ in sorted_words if word not in technical_skills][:10]

    for category, skills in top_technical_skills.items():
        print(f"Top 10 Technical Skills for {category}: {', '.join(skills)}")

    for category, skills in top_soft_skills.items():
        print(f"Top 10 Soft Skills for {category}: {', '.join(skills)}")

def generate_soft_skill(job_category, job_descriptions, soft_skills):
    # Initialize a dictionary to store relevant skills for each soft skill
    relevant_skills_dict = {}

    for skill in soft_skills:
        # Combine the job descriptions for the specified job category
        category_descriptions = job_descriptions[job_descriptions[job_category] == 1]['Job_Description']
        all_descriptions = " ".join(category_descriptions)

        # Tokenize the combined string
        words = word_tokenize(all_descriptions)

        # Define stop words
        stop_words = set(stopwords.words('english'))

        # Remove stop words and duplicate words
        filtered_words = [word for word in words if word.lower() not in stop_words]
        unique_words = list(set(filtered_words))

        # Rejoin the unique words into a single string
        processed_description = " ".join(unique_words)

        # Load the processed description into spaCy
        doc = nlp(processed_description)

        # Access individual tokens
        tokens = [token.text for token in doc]

        # Get word vectors for each token
        word_vectors = [token.vector for token in doc]

        # Convert the list of word vectors to a NumPy array
        word_vectors_array = np.array(word_vectors)

        # Initialize a dictionary to store soft skill vectors
        soft_skill_vectors = {}
        skill_doc = nlp(skill)
        soft_skill_vectors[skill] = skill_doc.vector

        # Create a helper function to find top n related soft skills
        def get_top_tokens_for_skill(skill_vector, word_vectors, n=8):
            # Calculate cosine similarity between the skill vector and all word vectors
            similarities = cosine_similarity([skill_vector], word_vectors)[0]

            # Sort the tokens by similarity and get the top N
            top_tokens = [token for token, similarity in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:n]]

            return top_tokens

        # List of all tokens (from the processed description)
        all_tokens = [token.text for token in doc]

        # Create a list to store the top N tokens for the current skill
        top_tokens_for_skill = get_top_tokens_for_skill(skill_doc.vector, word_vectors_array, n=10)

        # Store the relevant skills for the current soft skill in the dictionary
        relevant_skills_dict[skill] = [all_tokens[token] for token in top_tokens_for_skill]

    return relevant_skills_dict

#Now we can define a function to find soft skills belong to each job category.
def calculate_soft_skill_occurrences(job_category, job_descriptions, soft_skills):
    # Generate a dictionary of relevant skills for each soft skill
    relevant_skills_dict = generate_soft_skill(job_category, job_descriptions, soft_skills)

    # Combine the job descriptions for the specified job category
    category_descriptions = job_descriptions[job_descriptions[job_category] == 1]['Job_Description']
    all_descriptions = " ".join(category_descriptions)

    # Tokenize the combined string
    words = word_tokenize(all_descriptions)

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words and duplicate words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    unique_words = list(set(filtered_words))

    # Rejoin the unique words into a single string
    processed_description = " ".join(unique_words)

    # Load the processed description into spaCy
    doc = nlp(processed_description)

    # Create a dictionary to store skill occurrences
    skill_occurrences = {skill: 0 for skill in soft_skills}

    for skill in soft_skills:
        # Include the main skill itself
        skill_occurrences[skill] = all_descriptions.lower().count(skill.lower())

        # Include the occurrences of relevant skills
        if skill in relevant_skills_dict:
            relevant_skills = relevant_skills_dict[skill]
            for relevant_skill in relevant_skills:
                skill_occurrences[skill] += all_descriptions.lower().count(relevant_skill.lower())

    return skill_occurrences

def calculate_tech_skill_occurrences(job_category, job_descriptions, tech_skills):
    # Combine the job descriptions for the specified job category
    category_descriptions = job_descriptions[job_descriptions[job_category] == 1]['Job_Description']
    all_descriptions = " ".join(category_descriptions)

    # Tokenize the combined string
    words = word_tokenize(all_descriptions)

    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Remove stop words and duplicate words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    unique_words = list(set(filtered_words))

    # Rejoin the unique words into a single string
    processed_description = " ".join(unique_words)

    # Create a Counter to store skill occurrences
    skill_occurrences = Counter()

    for skill in tech_skills:
        # Use regular expressions to find exact matches with word boundaries
        pattern = r'\b{}\b'.format(re.escape(skill))
        matches = re.findall(pattern, all_descriptions, flags=re.IGNORECASE)

        # Count the occurrences of the technical skill
        skill_occurrences[skill] = len(matches)

    return skill_occurrences

def min_max_scaling(occurrences):
    # Find the minimum and maximum occurrences
    min_count = min(occurrences.values())
    max_count = max(occurrences.values())

    scaled_occurrences = {}

    # Apply Min-Max scaling
    for skill, count in occurrences.items():
        scaled_count = (count - min_count) / (max_count - min_count)
        scaled_occurrences[skill] = scaled_count

    return scaled_occurrences


def generate_word_clouds(job_category, job_descriptions, soft_skills, tech_skills, tech_skill_multiplier=2):
    soft_skill_occurrences = calculate_soft_skill_occurrences(job_category, job_descriptions, soft_skills)
    soft_skill_frequency_dict = dict(soft_skill_occurrences)

    tech_skill_occurrences = calculate_tech_skill_occurrences(job_category, job_descriptions, tech_skills)
    tech_skill_frequency_dict = dict(tech_skill_occurrences)

    # Create a word cloud for soft skills
    soft_skills_wordcloud = WordCloud(width=400, height=200, background_color='white')
    soft_skills_wordcloud.generate_from_frequencies(soft_skill_frequency_dict)

    # Create a word cloud for technical skills
    tech_skills_wordcloud = WordCloud(width=400, height=200, background_color='white')
    tech_skills_wordcloud.generate_from_frequencies(tech_skill_frequency_dict)

    # Combine soft and emphasized technical skills for a combined word cloud
    combined_skills = {**soft_skill_frequency_dict, **{skill: count * tech_skill_multiplier for skill, count in tech_skill_frequency_dict.items()}}
    scaled_combined_skills = min_max_scaling(combined_skills)

    # Create a word cloud for all skills combined
    all_skills_wordcloud = WordCloud(width=800, height=400, background_color='white')
    all_skills_wordcloud.generate_from_frequencies(scaled_combined_skills)

    # Define mosaic layout for subplots
    subplot_mosaic = {
        'AA': ['soft', 'tech'],
        'AB': 'combined'
    }

    # Create subplots using subplot_mosaic
    fig, axs = plt.subplot_mosaic(subplot_mosaic, figsize=(12, 8))

    # Display word clouds in subplots
    axs['soft'].imshow(soft_skills_wordcloud, interpolation='bilinear')
    axs['soft'].set_title("Soft Skills Word Cloud")
    axs['soft'].axis('off')

    axs['tech'].imshow(tech_skills_wordcloud, interpolation='bilinear')
    axs['tech'].set_title("Tech Skills Word Cloud")
    axs['tech'].axis('off')

    axs['combined'].imshow(all_skills_wordcloud, interpolation='bilinear')
    axs['combined'].set_title("All Skills Word Cloud")
    axs['combined'].axis('off')

    # Show the figure with subplots
    plt.tight_layout()

    plt.show()

# Example usage:
# Assuming 'job_category', 'job_descriptions', 'soft_skills', 'tech_skills' are available
# generate_word_clouds(job_category, job_descriptions, soft_skills, tech_skills)
vectorizer = TfidfVectorizer(
    max_features=1000,
    stop_words='english',
    lowercase=True,
    token_pattern=r'\w{3,}'
)

def preprocess_data(jda_clean, job_categories):

    X = vectorizer.fit_transform(jda_clean['Job_Description'])
    y = jda_clean[job_categories]
    return X, y

def train_binary_relevance(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
    br_clf = MultiOutputClassifier(MultinomialNB())
    br_clf.fit(X_train, y_train)
    y_predicted = br_clf.predict(X_test)
    accuracy = br_clf.score(X_test, y_test)
    print("Binary Relevance Model accuracy:", accuracy)
    return y_test, y_predicted

def train_multilabel_classifier(X, y, job_categories):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    multioutput_classifier = MultiOutputClassifier(rf_classifier)
    multioutput_classifier.fit(X_train, y_train)
    y_pred = multioutput_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Multilabel Classifier Model accuracy:", accuracy)
    report = classification_report(y_test, y_pred, target_names=job_categories, zero_division=0)
    print("Classification Report for Multilabel Classifier:\n", report)
    return y_test, y_pred

def train_xgboost_classifier(X, y, job_categories):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
    param_grid = {
        'max_depth': [3, 5, 7],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    xgb_classifier = XGBClassifier(random_state=42)
    grid_search = GridSearchCV(xgb_classifier, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_xgb_classifier = XGBClassifier(**best_params, random_state=42)
    best_xgb_classifier.fit(X_train, y_train)
    y_pred = best_xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("XGBoost Classifier Model accuracy:", accuracy)
    report = classification_report(y_test, y_pred, target_names=job_categories, zero_division=0)
    print("Classification Report for XGBoost Classifier:\n", report)
    return y_test, y_pred

# Example usage:
X, y = preprocess_data(jda_clean, job_categories)

# Binary Relevance
y_test_br, y_pred_br = train_binary_relevance(X, y)

# Multilabel Classifier
y_test_mc, y_pred_mc = train_multilabel_classifier(X, y, job_categories)

# XGBoost Classifier
y_test_xgb, y_pred_xgb = train_xgboost_classifier(X, y, job_categories)



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def train_neural_network(jda_clean, job_categories):
    X = vectorizer.fit_transform(jda_clean['JD_Trans']).toarray()
    y = jda_clean[job_categories]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = keras.Sequential()
    model.add(layers.Input(shape=(X.shape[1],)))  # Input layer with the number of input features
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='sigmoid'))  # Output layer with 5 units

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model


def train_and_save_text_cnn(jda, job_categories, save_weights_path):
    # Define X as the job descriptions (JD_Trans)
    X = jda['Job_Description']
    # Define y as the one-hot encoded job categories
    y = jda[job_categories]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)

    train_texts = X_train
    test_texts = X_test

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_texts)
    X_train = tokenizer.texts_to_sequences(train_texts)
    X_test = tokenizer.texts_to_sequences(test_texts)

    X_train = pad_sequences(X_train, maxlen=100)
    X_test = pad_sequences(X_test, maxlen=100)

    num_labels = 5
    CNN_model = Sequential()
    CNN_model.add(Embedding(input_dim=5000, output_dim=100, input_length=100))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(128, activation='relu'))
    CNN_model.add(Dense(num_labels, activation='sigmoid'))

    CNN_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    CNN_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model's weights
    CNN_model.save('trained_model.h5')

def predict_using_saved_model(data_to_predict, model_path):
    loaded_model = load_model(model_path)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data_to_predict)
    sequences = tokenizer.texts_to_sequences(data_to_predict)
    padded_sequences = pad_sequences(sequences, maxlen=100)

    predictions = loaded_model.predict(padded_sequences)
    return predictions
