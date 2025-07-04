from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Cargar modelo
modelo_path = os.path.join(os.path.dirname(__file__), 'boosting.sav')
modelo = pickle.load(open(modelo_path, 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    if request.method == 'POST':
        try:
            data = [
                int(request.form['Pregnancies']),
                int(request.form['Glucose']),
                int(request.form['BloodPressure']),
                int(request.form['SkinThickness']),
                int(request.form['Insulin']),
                float(request.form['BMI']),
                float(request.form['DiabetesPedigreeFunction']),
                int(request.form['Age'])
            ]
            input_data = np.array([data])
            prediccion = modelo.predict(input_data)[0]
            resultado = 'Tiene diabetes' if prediccion == 1 else 'No tiene diabetes'
        except Exception as e:
            resultado = f'Error: {e}'
    return render_template('index.html', resultado=resultado)

if __name__ == '__main__':
    app.run(debug=True)
