from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import json
import subprocess
from collections import OrderedDict
import math



app = Flask(__name__, template_folder="templates")

@app.route("/")

# Function to process and format the salary data


def index():
    job_category, formatted_salaries, formatted_locations, actual_percentages, log_scaled_widths, total_jobs = load_and_format_job_data()
    return render_template('index.html', job_percentages=job_category, formatted_salaries=formatted_salaries, formatted_locations=formatted_locations, actual_percentages=actual_percentages, log_scaled_widths=log_scaled_widths, total_jobs=total_jobs )



@app.route('/', methods=['GET', 'POST'])
def predict():
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
        print(f"Processed data: {predicted_results}")
        job_category, formatted_salaries = load_and_format_job_data()

        return render_template('index.html', prediction=formatted_prediction, job_percentages=job_category, formatted_salaries=formatted_salaries)


    return render_template('index.html')

def load_and_format_job_data():
    with open('static/EDA_result.json', 'r') as file:
        job_data = json.load(file)

    job_category = job_data.get('Job_Category')
    est_salary = job_data.get('Est_Salary')
    formatted_salaries = format_salaries(est_salary)
    job_location = job_data.get('City_Job_Count')
    formatted_locations, actual_percentages, log_scaled_widths, total_jobs = format_job_locations(job_location)

    return job_category, formatted_salaries, formatted_locations, actual_percentages, log_scaled_widths, total_jobs

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




if __name__ == '__main__':
        app.run(host="localhost", port=8080, debug=True)
