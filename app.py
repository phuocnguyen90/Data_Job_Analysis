from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import json
import subprocess


app = Flask(__name__, template_folder="templates")

@app.route("/")



def index():
    Job_Category, Est_Salary = load_job_percentages()
    return render_template('index.html', job_percentages=Job_Category, sal_count=Est_Salary )

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
        Job_Category, Est_Salary = load_job_percentages()

        return render_template('index.html', prediction=formatted_prediction, job_percentages=Job_Category, sal_count=Est_Salary )

    return render_template('index.html')

def load_job_percentages():
    with open('static/EDA_result.json', 'r') as file:
        job_percentages_dict = json.load(file)

    Job_Category = job_percentages_dict.get('Job_Category')
    Est_Salary = job_percentages_dict.get('Est_Salary')

    return Job_Category, Est_Salary

if __name__ == '__main__':
        app.run(host="localhost", port=8080, debug=True)
