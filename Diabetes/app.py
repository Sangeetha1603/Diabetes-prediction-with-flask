from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_diabetes():
    Pregnancies = int(request.form.get('Pregnancies',0))
    Glucose= int(request.form.get('Glucose'))
    BloodPressure = int(request.form.get('BloodPressure'))
    SkinThickness = int(request.form.get('SkinThickness'))
    Insulin = int(request.form.get('Insulin'))
    BMI = float(request.form.get('BMI'))
    DiabetesPedigreeFunction= float(request.form.get('DiabetesPedigreeFunction'))
    Age=int(request.form.get('Age'))

    result = model.predict(np.array([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]).reshape(1,8))
    if result[0] == 0:
        result = 'The person is diabetic'
    else:
        result = 'The person is not diabetic'

    return render_template('index.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
