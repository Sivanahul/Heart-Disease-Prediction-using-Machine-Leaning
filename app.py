import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# Load the dataset
dataset = pd.read_csv(r"C:\Heart_disease\heart.csv")

# Split data into predictors and target
predictors = dataset.drop("target", axis=1)
target = dataset["target"]

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Train Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, Y_train)

# Function to predict heart disease presence or absence
def predict_heart_disease_presence(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
    prediction = rfc.predict(input_data)
    return prediction[0]

# Streamlit interface
st.title("Heart Disease Prediction")
st.write("Enter the required information to predict the presence of heart disease.")

age = st.number_input("Age", min_value=0, max_value=120, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
fbs = 1 if fbs == "Yes" else 0
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=150)
exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
exang = 1 if exang == "Yes" else 0
oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", min_value=0.0, max_value=10.0, value=0.0)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

if st.button("Predict"):
    prediction = predict_heart_disease_presence(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    st.write("Heart Disease Presence:", "Yes" if prediction == 1 else "No")
