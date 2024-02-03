from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import json
from collections import OrderedDict
import math

def load_job_data(file_path):
    """
    Load job data from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    dict: The job data loaded from the file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the file: {file_path}")
        return {}


def format_salaries(sal_count):
    salary_map = OrderedDict([
        ("0-500", "$0 - $500"),
        ("500-1500", "$500 - $1500"),
        ("1500-3000", "$1500 - $3000"),
        ("3000-5000", "$3000 - $5000"),
        ("More than 5000", "More than $5000")
    ])

    total_count = sum(sal_count.values())

    processed_salaries = OrderedDict()
    for bin in salary_map.keys():
        count = sal_count.get(bin, 0)  # Get count for each bin, default to 0 if not found
        percentage = (count / total_count) * 100 if total_count > 0 else 0  # Avoid division by zero
        processed_salaries[bin] = {
            'formatted': salary_map[bin],
            'percentage': round(percentage, 2),
            'count': count
        }

    return processed_salaries

def format_job_locations(job_location, exclude_location='VN'):

    # Remove the specified location
    if exclude_location in job_location:
        excluded_count = job_location.pop(exclude_location)
    else:
        excluded_count = 0

    # Sort locations by job count and select the top 9
    sorted_locations = sorted(job_location.items(), key=lambda x: x[1], reverse=True)
    top_locations = sorted_locations[:9]

    # Sum the counts of all other locations plus the excluded location count
    other_count = sum(count for loc, count in sorted_locations[9:]) + excluded_count

    formatted_locations = OrderedDict()
    for loc, count in top_locations:
        formatted_locations[loc] = count

    # Add the "Other" category
    if other_count > 0:
        formatted_locations['Other'] = other_count

    total_jobs = sum(formatted_locations.values())
    actual_percentages = {loc: (count / total_jobs * 100) for loc, count in formatted_locations.items()}


    # Find the maximum log count for scaling
    max_log_count = math.log(max(formatted_locations.values(), default=1))
    # Calculate the logarithmic scale for the width of each bar
    log_scaled_widths = {}


    for loc, count in formatted_locations.items():
        if count > 0:
            log_count = math.log(count)
            scaled_width = (log_count / max_log_count) * 100
        else:
            scaled_width = 0
        log_scaled_widths[loc] = round(scaled_width, 2)

    return formatted_locations, actual_percentages, log_scaled_widths, total_jobs


def process_job_data(file_path='static/EDA_result.json'):
    """
    Process job data from the JSON file and format it for presentation.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    dict: Dictionary containing formatted job category, salaries, locations, percentages, and total jobs.
    """
    job_data = load_job_data(file_path)

    job_category = job_data.get('Job_Category')
    est_salary = job_data.get('Est_Salary')
    formatted_salaries = format_salaries(est_salary)
    job_location = job_data.get('City_Job_Count')
    formatted_locations, actual_percentages, log_scaled_widths, total_jobs = format_job_locations(job_location)

    return {
        'job_category': job_category,
        'formatted_salaries': formatted_salaries,
        'formatted_locations': formatted_locations,
        'actual_percentages': actual_percentages,
        'log_scaled_widths': log_scaled_widths,
        'total_jobs': total_jobs
    }

def process_prediction(user_input):

    if request.method == 'POST':
        job_title = request.form['job_title']
        job_description = request.form['job_description']
        yoe = request.form['yoe']
        est_salary = request.form['est_salary']

        print(f"Received user input: {job_title}")
        
        # Call the get_category function directly with user inputs
        predicted_results = get_category(job_description, job_title, yoe, est_salary)
        
        # Format the prediction into a human-readable format
        formatted_prediction = format_prediction(predicted_results)


        return formatted_prediction
