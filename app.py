from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import json
import subprocess
from collections import OrderedDict



app = Flask(__name__, template_folder="templates")

@app.route("/")

# Function to process and format the salary data


def index():
    job_category, formatted_salaries = load_and_format_job_percentages()
    return render_template('index.html', job_percentages=job_category, formatted_salaries=formatted_salaries)



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
        job_category, formatted_salaries = load_and_format_job_percentages()

        return render_template('index.html', prediction=formatted_prediction, job_percentages=job_category, formatted_salaries=formatted_salaries)


    return render_template('index.html')

def load_and_format_job_percentages():
    with open('static/EDA_result.json', 'r') as file:
        job_percentages_dict = json.load(file)

    job_category = job_percentages_dict.get('Job_Category')
    est_salary = job_percentages_dict.get('Est_Salary')
    formatted_salaries = format_salaries(est_salary)

    return job_category, formatted_salaries

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



if __name__ == '__main__':
        app.run(host="localhost", port=8080, debug=True)
