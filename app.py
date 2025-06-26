from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the best model
model = joblib.load('Notebook_file/random_forest_best_model.joblib')  # Change filename if needed

# Example: List your feature names in the order expected by the model
feature_names = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get values from form and convert to float
        input_features = [float(request.form[feature]) for feature in feature_names]
        # Reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)
        # Predict
        prediction = model.predict(input_array)[0]
    return render_template('index.html', prediction=prediction, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
    