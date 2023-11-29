import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import re
import string
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

df=config.df

def plot_salary_distribution_by_experience(dataframe, action='show', cut_off_salary=None):
    # Define bin intervals
    bins = [0, 1, 2, 5, 7, 10, float('inf')]
    dataframe['YOE_Bin'] = pd.cut(dataframe['YOE'], bins=bins, labels=[f'{bins[i]}-{bins[i+1]}' for i in range(len(bins) - 1)], right=False)

    if cut_off_salary is not None:
        dataframe = dataframe[dataframe['Est_Salary'] <= cut_off_salary]

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
    box_plot = plt.boxplot(boxplot_data, labels=dataframe['YOE_Bin'].cat.categories, patch_artist=True, showfliers=False, meanline =False)

    # Apply colors to the boxes (using different colors for mean and median)
    box_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']

    for patch, color in zip(box_plot['boxes'], box_colors):
        patch.set_facecolor(color)

    # Calculate and add mean value labels to each box
    for i, data in enumerate(boxplot_data):
        mean_salary = int(round(np.mean(data)))
        plt.text(i+1, mean_salary, mean_salary, horizontalalignment='center', verticalalignment='bottom', color='blue')

    # Add median value labels to each box with orange color
    for i, median_val in enumerate(median_values):
        plt.text(i+1, median_val, int(median_val), horizontalalignment='center', verticalalignment='top', color='#FFA500')

    # Create a legend for mean and median colors
    mean_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Mean')
    median_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFA500', markersize=10, label='Median')

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

plot_salary_distribution_by_experience(df,action='save',cut_off_salary=10000)    