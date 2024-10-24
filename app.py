import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
DATASET_PATH = r"C:\Users\HP\Desktop\Crop_recommendation.csv" # Replace with your dataset path

# Load and preview dataset
data = pd.read_csv(DATASET_PATH)

# Streamlit UI
st.title("Crop Recommender App")
st.write("This app helps farmers select the best crop based on soil and environmental conditions.")

# Preview the dataset
st.write("Dataset Preview:")
st.write(data.head())

# Data preprocessing
X = data.drop(columns=['label'])  # Features
y = data['label']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Display accuracy on test data
accuracy = model.score(X_test, y_test)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input for prediction
st.sidebar.title("Input Soil & Environmental Conditions")
nitrogen = st.sidebar.slider("Nitrogen Content (N)", 0, 140)
phosphorus = st.sidebar.slider("Phosphorus Content (P)", 5, 145)
potassium = st.sidebar.slider("Potassium Content (K)", 5, 205)
temperature = st.sidebar.slider("Temperature (°C)", 8.0, 45.0)
humidity = st.sidebar.slider("Humidity (%)", 10.0, 100.0)
ph = st.sidebar.slider("Soil pH", 3.5, 10.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0)

# Create a dataframe from user input
user_input = pd.DataFrame({
    'N': [nitrogen],
    'P': [phosphorus],
    'K': [potassium],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})

st.write("User Input Conditions:")
st.write(user_input)

# Make predictions
prediction = model.predict(user_input)
st.write(f"Recommended Crop: {prediction[0]}")