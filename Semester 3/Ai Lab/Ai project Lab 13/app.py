from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            features = [
                float(request.form.get("tests_per_case")),
                float(request.form.get("reproduction_rate")),
                float(request.form.get("stringency_index")),
                float(request.form.get("population_density")),
                float(request.form.get("median_age")),
                float(request.form.get("aged_65_older")),
                float(request.form.get("aged_70_older")),
                float(request.form.get("gdp_per_capita")),
                float(request.form.get("extreme_poverty")),
                float(request.form.get("cardiovasc_death_rate")),
                float(request.form.get("diabetes_prevalence")),
                int(request.form.get("date")),
                int(request.form.get("continent")),
                int(request.form.get("location")),
                float(request.form.get("positive_rate")),
            ]
            features_array = np.array([features])
            prediction = model.predict(features_array)
            return render_template("index.html", prediction_text=f"Predicted New Deaths: {int(prediction[0])}")
        except Exception as e:
            return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
