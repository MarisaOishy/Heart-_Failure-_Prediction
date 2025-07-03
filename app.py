from flask import Flask, request, render_template, flash
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')  # For flash messages

# Load the best model
try:
    model = joblib.load('Notebook_file/random_forest_best_model.joblib')
except Exception as e:
    model = None
    print(f"Model loading failed: {e}")

# List your feature names in the order expected by the model
feature_names = [
    'age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets',
    'serum_creatinine', 'serum_sodium', 'time',
    'anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'
]

def validate_input(form, feature_names):
    errors = []
    values = []
    for feature in feature_names:
        value = form.get(feature, '').strip()
        if value == '':
            errors.append(f"{feature.replace('_', ' ').title()} is required.")
            continue
        try:
            values.append(float(value))
        except ValueError:
            errors.append(f"{feature.replace('_', ' ').title()} must be a number.")
    return values, errors

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        input_features, errors = validate_input(request.form, feature_names)
        if errors:
            for error in errors:
                flash(error, 'danger')
        elif model is None:
            flash('Prediction model is not available. Please contact the administrator.', 'danger')
        else:
            input_array = np.array(input_features).reshape(1, -1)
            try:
                prediction = int(model.predict(input_array)[0])
            except Exception as e:
                flash(f'Prediction failed: {e}', 'danger')
    return render_template('index.html', prediction=prediction, feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)
