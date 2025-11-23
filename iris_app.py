import streamlit as st
import joblib
import numpy as np
import os

# show title
st.title("🌸 Iris Flower Classifier 🌸")
st.write("Move sliders → Click Predict → see species and confidence")

# safety: check artifacts
if not os.path.exists("iris_model.joblib") or not os.path.exists("iris_scaler.joblib"):
    st.error("Model files missing. Please run training script and push artifacts.")
    st.stop()

# cache load so app starts faster
@st.cache_resource
def load_artifacts():
    model = joblib.load("iris_model.joblib")
    scaler = joblib.load("iris_scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# sliders
sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = scaler.transform([[sl, sw, pl, pw]])
    # if model supports predict_proba:
    try:
        probs = model.predict_proba(data)[0]
        idx = int(np.argmax(probs))
        species = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"Predicted: {species[idx]} ({probs[idx]*100:.1f}% confidence)")
    except Exception:
        pred = model.predict(data)[0]
        species = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"Predicted: {species[int(pred)]} (confidence not available)")

# small footer: model info
st.markdown("---")
st.write("Model: SVM (scikit-learn) • Accuracy ≈ 96%")
st.write("Model version: v1")
