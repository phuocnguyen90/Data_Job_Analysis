import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from test import plot_salary_distribution_by_experience

# Import config from the parent directory
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

job_categories =config.job_categories
df = config.test

def generate_json_report(df, job_categories):
    if not set(job_categories).issubset(df.columns):
        raise ValueError("Job category columns not found in the DataFrame.")

    category_percentages = {}
    total_records = len(df)
    # Calculate job category percentages
    for category in job_categories:
        category_count = df[category].sum()
        category_percentage = (category_count / total_records) * 100
        category_percentages[category] = int(category_percentage)

    # Calculate salary range counts
    bins = [0, 500, 1500, 3000, 5000, float('inf')]
    labels = ['0-500', '500-1500', '1500-3000', '3000-5000', 'More than 5000']

    df['Est_Salary'] = pd.cut(df['Est_Salary'], bins=bins, labels=labels, include_lowest=True)
    salary_counts = df['Est_Salary'].value_counts().to_dict()

    # Calculate job count per city
    city_counts = df['Location'].value_counts().to_dict()

    job_percentages = {
        'Job_Category': category_percentages,
        'Est_Salary': salary_counts,
        'City_Job_Count': city_counts
    }

    # Convert dictionary to JSON
    job_percentages_json = json.dumps(job_percentages)

    # Write JSON to a file
    with open('static/EDA_result.json', 'w') as file:
        file.write(job_percentages_json)


# Call the function and handle the potential error
try:
    # Assuming df and job_categories are defined somewhere
    generate_json_report(df, job_categories)
except ValueError as e:
    print(f"Error: {e}")

