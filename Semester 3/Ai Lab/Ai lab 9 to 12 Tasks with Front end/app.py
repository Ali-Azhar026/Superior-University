import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['price']),
            int(request.form['yield']),
            int(request.form['production']),
            int(request.form['area_acres']),
            int(request.form['area_hectares']),
            int(request.form['total_value'])
        ]
        prediction = model.predict([features])
        return render_template('index.html', prediction_text=f'Predicted Value: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)

