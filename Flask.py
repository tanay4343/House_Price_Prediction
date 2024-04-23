from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("data.pkl", "rb"))

@app.route('/')
def index():
    location = sorted(data["location"].unique())
    return render_template('index.html', location=location)

@app.route('/Predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        location = str(request.form['location'])
        bhk = request.form['bhk']
        bath = request.form['bath']
        sqft = request.form['sqft']


    input_data = pd.DataFrame([[location, bhk, bath, sqft]], columns=['location', 'total_sqft', 'bath', 'bhk'])
    prediction = pipe.predict(input_data)[0] * 1e3
    rounded_prediction = round(prediction, 2)
    formatted_prediction = "{:.2f}".format(abs(rounded_prediction))

    return formatted_prediction


if __name__ == "__main__":
    app.run(debug=True)