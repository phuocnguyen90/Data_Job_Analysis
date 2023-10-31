from flask import Flask, render_template, request
from predict import get_category, format_prediction, category_names
import subprocess


app = Flask(__name__, template_folder="templates")

@app.route("/")

def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])

def predict():
    job_title = request.form['job_title']
    job_description = request.form['job_description']
    yoe = request.form['yoe']
    est_salary = request.form['est_salary']
    
    # Call the get_category function directly with user inputs
    predicted_results = get_category(job_description, job_title, yoe, est_salary)
    
    # Format the prediction into a human-readable format
    formatted_prediction = format_prediction(predicted_results)

    # Returning the formatted prediction as a string response
    return f"Predicted Category: {formatted_prediction}"


if __name__ == '__main__':
        app.run(host="0.0.0.0", port=50100, debug=True)
