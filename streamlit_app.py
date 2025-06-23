import streamlit as st
import numpy as np
import joblib

# Load models, scaler, and selected features
logreg_model = joblib.load("models/logistic_regression_model.pkl")
mlp_model = joblib.load("models/mlp_classifier_model.pkl")
scaler = joblib.load("models/scaler.pkl")
selected_features = joblib.load("models/selected_features.pkl")  # <- Loaded from model training

# Set up the page
st.set_page_config(page_title="Breast Cancer Predictor", layout="wide")
st.markdown("<h1 style='text-align: center; color: #e91e63;'>🔬 Breast Cancer Predictor</h1>", unsafe_allow_html=True)
st.markdown("Enter the medical details below:")

# Sidebar info
with st.sidebar:
    st.image("https://logodix.com/logo/1094286.png", width=120)
    st.markdown("**App Info**")
    st.write("This app uses two ML models to predict whether a tumor is **benign** or **malignant**.")
    st.markdown("- Logistic Regression\n- MLP Classifier (Neural Net)")
    st.write("Ensure values are realistic (based on real-world data).")

# Known ranges for feature sliders
feature_ranges = {
    "mean radius": (6.0, 30.0),
    "mean texture": (8.0, 40.0),
    "mean perimeter": (40.0, 200.0),
    "mean area": (140.0, 2500.0),
    "mean smoothness": (0.05, 0.2),
    "mean compactness": (0.01, 1.5),
    "mean concavity": (0.0, 0.5),
    "mean concave points": (0.0, 0.3),
    "worst concave points": (0.0, 0.5)
}

# Input layout
col1, col2 = st.columns(2)
input_data = []

for i, feature in enumerate(selected_features):
    col = col1 if i % 2 == 0 else col2
    min_val, max_val = feature_ranges.get(feature, (0.0, 1.0))  # default range fallback
    val = col.slider(
        f"{feature.replace('_', ' ').title()} ({min_val}–{max_val})",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2)
    )
    input_data.append(val)

# Prediction logic
st.markdown("---")
if st.button("🔍 Predict Cancer Type"):
    input_array = np.array([input_data])
    input_scaled = scaler.transform(input_array)

    pred_logreg = logreg_model.predict(input_scaled)[0]
    pred_mlp = mlp_model.predict(input_scaled)[0]

    prob_logreg = logreg_model.predict_proba(input_scaled)[0][1]
    prob_mlp = mlp_model.predict_proba(input_scaled)[0][1]

    st.subheader("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔹 Logistic Regression")
        st.metric(
            label="Prediction",
            value="🔴 Malignant" if pred_logreg else "🟢 Benign",
            delta=f"{prob_logreg*100:.2f}%" if pred_logreg else f"-{(1-prob_logreg)*100:.2f}%",
            delta_color="inverse" if pred_logreg else "normal"
        )

    with col2:
        st.subheader("🔹 MLP Classifier")
        st.metric(
            label="Prediction",
            value="🔴 Malignant" if pred_mlp else "🟢 Benign",
            delta=f"{prob_mlp*100:.2f}%" if pred_mlp else f"-{(1-prob_mlp)*100:.2f}%",
            delta_color="inverse" if pred_mlp else "normal"
        )

    st.success("✔ Prediction complete! Use values based on real patient data for best results.")
