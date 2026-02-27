import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="Stockout AI Suite | Strategic Replenishment",
    page_icon="📦",
    layout="wide"
)

# Professional UI Styling for Dark/Mixed Mode
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #3e4259;
    }
    div[data-testid="stMetricValue"] { color: #ffffff; }
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD PREDICTION ENGINE (PIPELINE)
# Calibrated for 14-day strategic replenishment (AUC 0.91)
MODEL_PATH = '04_Models/full_pipeline_14day_strategic.pkl'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Ensure the .pkl file exists in {MODEL_PATH}")

# 3. SIDEBAR (SIMULATION PANEL)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=100)
st.sidebar.title("Simulation Panel")
st.sidebar.markdown("Adjust parameters to simulate stockout risk in real-time.")

with st.sidebar:
    st.subheader("📦 Inventory Levels")
    inv_level = st.slider("Current Stock (Units)", 0, 1000, 450)
    units_sold = st.slider("Units Sold (Last 24h)", 0, 150, 30)
    
    st.subheader("💰 Commercial Strategy")
    price = st.number_input("Our Price ($)", value=150.0)
    comp_price = st.number_input("Competitor Price ($)", value=145.0)
    discount = st.slider("Applied Discount (%)", 0.0, 0.5, 0.1)
    
    st.subheader("🌍 Logistics Context")
    region = st.selectbox("Region", ["North", "South", "East", "West", "Central"])
    category = st.selectbox("Category", ["Electronics", "Fashion", "Home", "Toys", "Groceries"])
    is_weekend = st.checkbox("Is it a Weekend?")

# 4. MAIN DASHBOARD HEADER
st.title("📦 Strategic Stockout Early Warning System")
st.markdown(f"**Target Window:** 14-Day Strategic Replenishment | **Model Status:** Calibrated (AUC 0.91)")
st.markdown("---")

# 4. PREPARE INPUT (Labels corrected for Model compatibility)
input_df = pd.DataFrame({
    'Store ID': ['STR_PROD_99'],
    'Product ID': ['PROD_FINAL_CHECK'],
    'Category': [category],
    'Region': [region],
    'Weather Condition': ['Clear'],
    'Holiday/Promotion': ['None'],
    'Seasonality': ['Regular'],
    'Month': ['2'],
    'Day of Week': ['3'],
    'Inventory Level': [float(inv_level)],
    'Units Sold': [float(units_sold)],
    'Price': [float(price)],
    'Discount': [float(discount)],
    'Competitor Pricing': [float(comp_price)],
    'Is Weekend': [1 if is_weekend else 0]
})

# Cast as string for the model's encoder
cat_cols = ['Store ID', 'Category', 'Region', 'Weather Condition', 'Holiday/Promotion', 'Seasonality', 'Month', 'Day of Week', 'Product ID']
for col in cat_cols:
    input_df[col] = input_df[col].astype(str)
    
# 5. INFERENCE & ERROR HANDLING
# Converting prob to a native Python float to satisfy st.progress()
prob_raw = pipeline.predict_proba(input_df)[0][1]
prob = float(prob_raw) 

# 6. BUSINESS METRICS DISPLAY
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Risk Probability", value=f"{prob*100:.1f}%")

with col2:
    if prob > 0.75:
        status = "🚨 CRITICAL"
    elif prob > 0.40:
        status = "⚠️ WARNING"
    else:
        status = "✅ SAFE"
    st.metric(label="Inventory Health", value=status)

with col3:
    # Revenue at Risk = Probability * Price * Sales Momentum
    financial_impact = prob * price * units_sold
    st.metric(label="Revenue at Risk (14d)", value=f"${financial_impact:,.2f}")

# 7. RISK VISUAL ANALYSIS (Indestructible Progress Bar)
st.subheader("Safety Stock Analysis")

# Safeguard: Force value between 0.0 and 1.0 and ensure it's a standard float
safe_progress = float(np.clip(prob, 0.0, 1.0))
st.progress(safe_progress)

# 8. STRATEGIC RECOMMENDATION
st.markdown("---")
if prob > 0.75:
    st.error(f"**IMMEDIATE ACTION REQUIRED**: High stockout risk detected in **{region}**. We recommend issuing an international replenishment order immediately.")
elif prob > 0.40:
    st.warning(f"**WATCHLIST**: Imbalance detected between current sales velocity and stock. Monitor the **{category}** department closely.")
else:
    st.success("**HEALTHY INVENTORY**: Current stock levels are sufficient to cover the projected 14-day demand window.")

st.caption("Retail Stockout AI Suite v2.0 | MLOps End-to-End Certified")