
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

st.title("🔬 Breast Cancer Prediction - Logistic Regression")

st.write("Upload a Breast Cancer Dataset (CSV) or use the default dataset.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("No file uploaded. Please upload a dataset to proceed.")
    st.stop()

st.subheader("Dataset Preview")
st.write(df.head())

# Features and Target
if 'diagnosis' not in df.columns:
    st.error("Dataset must contain a 'diagnosis' column (M = Malignant, B = Benign).")
    st.stop()

X = df.drop(columns=['diagnosis'])
y = df['diagnosis'].map({'M': 1, 'B': 0})

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
st.subheader("📊 Model Evaluation")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# Predict new sample
st.subheader("🔮 Try Prediction with New Data")
input_data = []
for i in range(X.shape[1]):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    new_pred = model.predict([input_data])
    result = "Malignant (Cancer)" if new_pred[0] == 1 else "Benign (Non-Cancer)"
    st.success(f"Prediction: {result}")
