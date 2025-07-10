from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model and label encoder
try:
    model = pickle.load(open('model_compatible.pkl', 'rb'))
    le = pickle.load(open('label_encoder.pkl', 'rb'))
    print("Model and label encoder loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Make sure to run train_model.py first to generate the model files.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        engine_cc = int(request.form['engine_cc'])
        mileage = float(request.form['mileage'])
        brand = request.form['brand']

        # Validate inputs
        if brand not in le.classes_:
            return render_template('index.html', 
                                 prediction_text=f"Error: Brand '{brand}' not supported. Available brands: {', '.join(le.classes_)}")

        brand_encoded = le.transform([brand])[0]

        input_data = np.array([[age, engine_cc, mileage, brand_encoded]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', 
                             prediction_text=f"Predicted Bike Price: ₹{prediction:,.2f}")
    
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}")

@app.route('/health')
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Use Railway's PORT environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
