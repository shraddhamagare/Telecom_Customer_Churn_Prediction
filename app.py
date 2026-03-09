import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page settings
st.set_page_config(page_title="Telecom Churn Dashboard", page_icon="📊", layout="wide")

# Load model and dataset
model = pickle.load(open("churn_model.pkl", "rb"))
df = pd.read_csv("churn_dataset.csv")

# Title
st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.markdown("Machine Learning model to predict telecom customer churn.")

st.markdown("---")

# Sidebar
st.sidebar.header("Customer Input")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)

# Prediction
if st.sidebar.button("Predict"):

    features = np.array([[tenure, monthly_charges]])
    prediction = model.predict(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

st.markdown("---")

# Dataset preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    churn_counts = df["Churn"].value_counts()
    st.bar_chart(churn_counts)

with col2:
    st.subheader("Monthly Charges Distribution")
    st.line_chart(df["MonthlyCharges"])

st.markdown("---")

st.subheader("Project Information")

st.write("""
This project predicts telecom customer churn using machine learning.

**Steps involved:**
- Data preprocessing
- Exploratory Data Analysis
- Model training (SVM, Logistic Regression, KNN, Decision Tree)
- Model comparison
- Streamlit web deployment
""")

st.caption("Developed using Python, Scikit-learn, Streamlit")


