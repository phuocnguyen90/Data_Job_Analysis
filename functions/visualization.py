import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
# Import config from the parent directory
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
sys.path.append(parent_dir)
import config 

def plot_job_category_counts(job_categories, dataframe):
    category_counts = []

    # Count the number of occurrences for each job category
    for category in job_categories:
        count = dataframe[category].sum()
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

def plot_salary_distribution_by_experience(dataframe, action='show'):
    # Define bin intervals 
    bins = [0, 1, 2, 5, 7, 10, float('inf')]
    dataframe['YOE_Bin'] = pd.cut(dataframe['YOE'], bins=bins, labels=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)], right=False)

    # Extract data for the box plot
    boxplot_data = [dataframe['Est_Salary'][dataframe['YOE_Bin'] == bin_label].dropna().values for bin_label in dataframe['YOE_Bin'].cat.categories]

    median_values = []
    for data in boxplot_data:
        if len(data) > 0:  
            median = np.median(data)
            median_values.append(median)
        else:
            median_values.append(np.nan)

    # Initialize the figure before creating the box plot
    plt.figure(figsize=(10, 15))

    # Create the box plot with custom colors
    box_plot = plt.boxplot(boxplot_data, labels=dataframe['YOE_Bin'].cat.categories, patch_artist=True)

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
    # Create the box plot with custom colors for median (orange) and mean (blue)

    plt.xticks(rotation=45)
    plt.title("Salary Distribution by Years of Experience")
    plt.xlabel("Years of Experience")
    plt.ylabel("Estimated Salary")
    

    if action == 'save':
        plt.savefig('static/salary_distribution_plot.png')  # Save the plot as an image in the 'static' folder
    elif action == 'show':
        plt.show()


def plot_top_locations(dataframe, n=15, save=False):
    location_counts = dataframe['Location'].value_counts()

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

    # Save the job count per location into a JSON file if save=True
    if save:
        output_file = 'job_counts_per_location.json'
        job_counts_dict = top_n_locations.to_dict()
        with open(output_file, 'w') as json_file:
            json.dump(job_counts_dict, json_file)

    # Show the bar chart
    plt.show()


def generate_job_description_word_cloud(dataframe):
    all_descriptions = " ".join(dataframe['Job_Description'])

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


    filtered_words = [word for word in filtered_words if word not in config.remove_words]

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title('Word Cloud of Most Frequent Words in Job Descriptions')
    plt.show()


def calculate_top_skills_per_category(dataframe, technical_skills, remove_words):
    top_technical_skills = {}
    top_soft_skills = {}

    categories = config.job_categories

    for category in categories:
        descriptions = dataframe[dataframe[category] == 1]['Job_Description']
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

# Create a helper function to find top n related soft skills
def get_top_tokens_for_skill(skill_vector, word_vectors, n=8):
    # Calculate cosine similarity between the skill vector and all word vectors
    similarities = cosine_similarity([skill_vector], word_vectors)[0]

    # Sort the tokens by similarity and get the top N
    top_tokens = [token for token, similarity in sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:n]]

    return top_tokens

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
        doc = config.nlp(processed_description)

        # Access individual tokens
        tokens = [token.text for token in doc]

        # Get word vectors for each token
        word_vectors = [token.vector for token in doc]

        # Convert the list of word vectors to a NumPy array
        word_vectors_array = np.array(word_vectors)

        # Initialize a dictionary to store soft skill vectors
        soft_skill_vectors = {}
        skill_doc = config.nlp(skill)
        soft_skill_vectors[skill] = skill_doc.vector

        # List of all tokens (from the processed description)
        all_tokens = [token.text for token in doc]

        # Create a list to store the top N tokens for the current skill
        top_tokens_for_skill = get_top_tokens_for_skill(skill_doc.vector, word_vectors_array, n=10)

        # Store the relevant skills for the current soft skill in the dictionary
        relevant_skills_dict[skill] = [all_tokens[token] for token in top_tokens_for_skill]

    return relevant_skills_dict

# Now we can define a function to find soft skills belong to each job category.
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
    doc = config.nlp(processed_description)

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