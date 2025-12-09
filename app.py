import streamlit as st
import pandas as pd
import requests
import os
import cloudpickle

MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0.0/pipe_execution.pkl"
MODEL_PATH = "pipe_execution.pkl"

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)

pipeline = load_pipeline()

st.title("Retail Stockout Risk Scoring")

uploaded_file = st.file_uploader("Upload a CSV with your store inventory data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview:", df.head())
    preds = pipeline.predict_proba(df)[:, 1]
    df["Stockout_Risk"] = preds
    st.write("Predictions:", df)
    st.download_button("Download Results", df.to_csv(index=False), "predictions.csv")
else:
    st.info("Upload a CSV to score stockout risk.")
