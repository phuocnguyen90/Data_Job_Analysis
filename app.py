from flask import Flask, render_template, request
from data_processing import process_job_data, process_prediction
from predict import get_category, format_prediction
import json

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = {
            'job_title': request.form['job_title'],
            'job_description': request.form['job_description'],
            'yoe': request.form['yoe'],
            'est_salary': request.form['est_salary']
        }
        prediction = process_prediction(user_input)
        job_data = process_job_data()
        job_data['prediction'] = prediction
        return render_template('index.html', **job_data)

    return render_template('index.html', **process_job_data())

if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True)
