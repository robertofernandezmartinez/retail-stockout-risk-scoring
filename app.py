import streamlit as st
import pandas as pd
import pickle
import requests
import os

# URL of the trained model in GitHub Releases
MODEL_URL = "https://github.com/robertofernandezmartinez/retail-stockout-risk-scoring/releases/download/v1.0/pipe_execution.pkl"
MODEL_PATH = "pipe_execution.pkl"

@st.cache_resource
def load_pipeline():
    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("Model loaded successfully!")

    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()

# Streamlit UI
st.set_page_config(page_title="Retail Stockout Risk Scoring", layout="wide")
st.title("🛒 Retail Stockout Risk Prediction")
st.write("Upload a CSV file with raw input data (same format as training).")

uploaded_file = st.file_uploader("📂 Upload inventory CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Rename columns (must match training format)
    rename_map = {
        "Date": "date",
        "Store ID": "store_id",
        "Product ID": "product_id",
        "Category": "category",
        "Region": "region",
        "Inventory Level": "inventory_level",
        "Units Sold": "units_sold",
        "Units Ordered": "units_ordered",
        "Demand Forecast": "demand_forecast",
        "Price": "price",
        "Discount": "discount",
        "Weather Condition": "weather_condition",
        "Holiday/Promotion": "holiday_promo",
        "Competitor Pricing": "competitor_pricing",
        "Seasonality": "seasonality",
    }
    df = df.rename(columns=rename_map)

    df['holiday_promo'] = df['holiday_promo'].astype("category")
    df['date'] = pd.to_datetime(df['date'])

    st.subheader("📊 Sample Preview")
    st.dataframe(df.head())

    # Model prediction
    pred_proba = pipeline.predict_proba(df)[:, 1]
    df["stockout_risk"] = pred_proba

    # Business impact metric
    df["economic_impact"] = df["stockout_risk"] * df["demand_forecast"] * df["price"]

    st.subheader("🔥 Predictions & Business Impact")
    st.dataframe(
        df[["store_id", "product_id", "stockout_risk", "economic_impact"]].head(20)
    )

    # Export option
    st.download_button(
        label="⬇️ Download Full Predictions",
        data=df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )

else:
    st.info("Please upload a file to begin.")
