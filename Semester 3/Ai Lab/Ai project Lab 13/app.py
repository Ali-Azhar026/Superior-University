import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

# Replace this list with your actual 26 input feature names
feature_names = [
    'tests_per_case', 'reproduction_rate', 'stringency_index', 'population_density', 'median_age',
    'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate',
    'diabetes_prevalence', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index',
    'icu_patients', 'icu_patients_per_million', 'hosp_patients', 'hosp_patients_per_million',
    'weekly_icu_admissions', 'weekly_hosp_admissions', 'date', 'continent', 'location', 'positive_rate',
    'people_vaccinated', 'people_fully_vaccinated'
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [float(request.form[feature]) for feature in feature_names]
        final_input = np.array(inputs).reshape(1, -1)
        prediction = model.predict(final_input)
        result = "Prediction: Death Likely (1)" if prediction[0] == 1 else "Prediction: No Death Likely (0)"
        return render_template('index.html', feature_names=feature_names, prediction_text=result)
    except Exception as e:
        return render_template('index.html', feature_names=feature_names, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)


