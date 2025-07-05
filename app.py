from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('astronomy_model.pkl') 

@app.route('/')
def home():
    heading = "Star Classification Using Astronomy Dataset"
    return render_template('index.html', heading=heading)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        u = float(request.form['u'])
        g = float(request.form['g'])
        r = float(request.form['r'])
        i = float(request.form['i'])
        z = float(request.form['z'])
        redshift = float(request.form['redshift'])

        features = np.array([[u, g, r, i, z, redshift]])
        
        prediction = model.predict(features)[0]
        print(f"Prediction raw value: {prediction}")  

        result = prediction

        return render_template('result.html', prediction=result)
    except Exception as e:
        return f"Error in prediction: {str(e)}. Please check your input."

if __name__ == '__main__':
    app.run(debug=True)
