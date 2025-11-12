import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.joblib")
scaler = joblib.load("iris_scaler.joblib")

st.title("🌸 Iris Flower Classifier 🌸")

sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
pw = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

if st.button("Predict"):
    data = scaler.transform([[sl, sw, pl, pw]])
    result = model.predict(data)[0]
    flower = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Flower: {flower[result]}")
