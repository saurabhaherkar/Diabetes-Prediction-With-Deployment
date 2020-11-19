import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)
model = joblib.load('diabetes_predictor.pkl')


@app.route('/')
def home():
    return render_template('index_diabetes.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    input_features = [float(i) for i in request.form.values()]
    feature_values = np.array(input_features)
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                     'DiabetesPedigreeFunction', 'Age']

    df = pd.DataFrame([feature_values], columns=feature_names)
    output = model.predict(df)
    if output == 1:
        return render_template('index_diabetes.html', prediction_text='OMG, Patient detected Diabetes. Take Care!!!')
    else:
        return render_template('index_diabetes.html', prediction_text='No worries, Patient not detected Diabetes '
                                                                      'anymore.')


if __name__ == '__main__':
    app.run(debug=True)