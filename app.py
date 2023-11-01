from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import subprocess


app = Flask(__name__, template_folder="templates")

@app.route("/")


def index():


    return render_template('index.html', job_percentages=job_percentages)

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

        return render_template('index.html', prediction=formatted_prediction, job_percentages=job_percentages )

    return render_template('index.html')
job_percentages = {
    'Data Analyst': 90,
    'Data Engineer': 80,
    'Data Scientist': 75,
    'Business Analyst': 50,
    'Business Intelligence': 50,
    'Others': 50
}


if __name__ == '__main__':
        app.run(host="0.0.0.0", port=50100, debug=True)
