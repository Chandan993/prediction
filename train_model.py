import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle

# Load data
df = pd.read_csv('bike_data.csv')

# Encode brand
le = LabelEncoder()
df['brand_encoded'] = le.fit_transform(df['brand'])

X = df[['age', 'engine_cc', 'mileage', 'brand_encoded']]
y = df['price']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model and label encoder
with open('bike_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model trained and saved.")
