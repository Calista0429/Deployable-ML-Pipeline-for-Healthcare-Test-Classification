import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from function import Pipeline  

# Load model Dockerfile
# model = joblib.load("./app/data/ensemble_model.pkl")
# cls_encoder = joblib.load("./app/data/categorical_encoder.pkl")
# label_encoder = joblib.load("./app/data/label_encoder.pkl")
# feature_names = joblib.load("./app/data/feature_names.pkl")

# FastAPI
model = joblib.load("data/ensemble_model.pkl")
cls_encoder = joblib.load("data/categorical_encoder.pkl")
label_encoder = joblib.load("data/label_encoder.pkl")
feature_names = joblib.load("data/feature_names.pkl")

st.set_page_config(page_title="Medical Test Prediction", layout="centered")
st.title("🧬 Medical Test Result Prediction")
st.markdown("""
Enter the patient details below to predict their medical test result.
This tool is powered by a machine learning model trained on medical records.
""")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This is a demo interface for predicting test results like **Abnormal**, **Normal**, or **Inconclusive** using structured medical data.")

# 表单输入
with st.form("prediction_form"):
    st.subheader("🔍 Patient Information")
    blood_type = st.selectbox("Blood Type", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
    medical_condition = st.selectbox("Medical Condition", ["Diabetes", "Cancer", "Asthma", "Hypertension", "Arthritis"])
    billing_amount = st.number_input("Billing Amount ($)", min_value=0.0, value=25000.0, step=100.0)
    age = st.slider("Age", min_value=0, max_value=100, value=50)
    stay_days = st.slider("Stay Days", min_value=0, max_value=60, value=10)

    submitted = st.form_submit_button("🔮 Predict")

if submitted:
    input_dict = {
        "Blood Type": blood_type,
        "Medical Condition": medical_condition,
        "Billing Amount": billing_amount,
        "Age": age,
        "Stay Days": stay_days
    }
    input_df = pd.DataFrame([input_dict])

    try:
        X_input = Pipeline(input_df, encoder=cls_encoder)
        X_input = X_input[feature_names]
        pred = model.predict(X_input)[0]
        label = label_encoder.inverse_transform([pred])[0]

        # Predict probabilities if possible
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0]
            labels = label_encoder.inverse_transform(np.arange(len(proba)))

            st.success(f"### ✅ Prediction: `{label}`")

            st.markdown("#### 📊 Prediction Confidence:")
            fig, ax = plt.subplots()
            ax.barh(labels, proba, color='skyblue')
            ax.set_xlim(0, 1)
            for i, v in enumerate(proba):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            st.pyplot(fig)
        else:
            st.success(f"### ✅ Prediction: `{label}`")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

