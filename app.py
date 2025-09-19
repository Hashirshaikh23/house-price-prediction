# app.py
import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model(path='models/best_model.pkl'):
    return joblib.load(path)

st.title("House Price Predictor (Boston)")

model_data = load_model('models/best_model.pkl')
cols = model_data['columns']

st.write("Enter feature values (defaults shown).")
inputs = {}
for c in cols:
    # choose a sensible default; use 0.0 if unknown
    example = 0.0
    inputs[c] = st.number_input(c, value=float(example))

if st.button("Predict"):
    df = pd.DataFrame([inputs], columns=cols)
    pred = model_data['model'].predict(df)
    st.success(f"Predicted MEDV: {pred[0]:.2f} (k$)")
