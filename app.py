from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and label encoder
model = pickle.load(open('bike_model.pkl', 'rb'))
le = pickle.load(open('label_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    engine_cc = int(request.form['engine_cc'])
    mileage = float(request.form['mileage'])
    brand = request.form['brand']

    brand_encoded = le.transform([brand])[0]

    input_data = np.array([[age, engine_cc, mileage, brand_encoded]])
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction_text=f"Predicted Bike Price: ₹{prediction:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
