from flask import Flask, render_template, request
import subprocess

app = Flask(__name__, template_folder="templates")

@app.route("/")

def index():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])

def predict():
    job_title = request.form['job_title']
    job_description = request.form['job_description']

    # Pass the job_title and job_description to predict.py using subprocess
    process = subprocess.Popen(['python', 'predict.py', job_title, job_description], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    predicted_salary = output.decode('utf-8').strip()

    return f"Predicted Salary: {predicted_salary}"

if __name__ == '__main__':
        app.run(host="0.0.0.0", port=50100, debug=True)
