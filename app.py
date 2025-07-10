from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Initialize model and label encoder as None
model = None
le = None

def load_or_train_model():
    """Load existing model or train new one if files don't exist"""
    global model, le
    
    try:
        # Try to load existing model
        model = pickle.load(open('model_compatible.pkl', 'rb'))
        le = pickle.load(open('label_encoder.pkl', 'rb'))
        print("✅ Model and label encoder loaded successfully!")
        return True
    except FileNotFoundError:
        print("⚠️ Model files not found. Training new model...")
        try:
            # Train new model if files don't exist
            import pandas as pd
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import LabelEncoder
            
            # Load and prepare data
            df = pd.read_csv('bike_data.csv')
            df['brand'] = df['brand'].str.strip()
            
            # Create and train model
            le = LabelEncoder()
            df['brand_encoded'] = le.fit_transform(df['brand'])
            
            X = df[['age', 'engine_cc', 'mileage', 'brand_encoded']]
            y = df['price']
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Save model files
            with open('model_compatible.pkl', 'wb') as f:
                pickle.dump(model, f)
            with open('label_encoder.pkl', 'wb') as f:
                pickle.dump(le, f)
            
            print("✅ New model trained and saved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False

# Load or train model on startup
load_or_train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, le
    
    if model is None or le is None:
        return render_template('index.html', 
                             prediction_text="❌ Error: Model not loaded. Please try again later.")
    
    try:
        age = int(request.form['age'])
        engine_cc = int(request.form['engine_cc'])
        mileage = float(request.form['mileage'])
        brand = request.form['brand']

        # Validate inputs
        if brand not in le.classes_:
            return render_template('index.html', 
                                 prediction_text=f"❌ Error: Brand '{brand}' not supported. Available brands: {', '.join(le.classes_)}")

        brand_encoded = le.transform([brand])[0]
        input_data = np.array([[age, engine_cc, mileage, brand_encoded]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', 
                             prediction_text=f"💰 Predicted Bike Price: ₹{prediction:,.2f}")
    
    except ValueError as e:
        return render_template('index.html', 
                             prediction_text="❌ Error: Please enter valid numbers for all fields.")
    except Exception as e:
        return render_template('index.html', 
                             prediction_text=f"❌ Error: {str(e)}")

@app.route('/health')
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
