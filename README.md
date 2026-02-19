# 🛒 Retail Stockout Risk Scoring & MLOps Pipeline

This project predicts **imminent retail stockout risks within a 72-hour window** and estimates the **economic impact** of potential stockouts. 

By applying a custom Business Logic layer to a trained XGBoost model, the system identifies products that require urgent replenishment without overwhelming store managers with alert fatigue.

## 🚀 Live Application

Try the interactive UI for business planners here 👇
👉 **https://retail-stockout-risk-scoring.streamlit.app/**

Upload a CSV containing daily store inventory data and receive:
- Stockout probability per product (72h timeframe).
- Business Impact Score estimation.
- A filtered, ranked table of the Top Critical Alerts ready for action.

## 📌 Executive Summary
Retailers operating with aggressive Just-In-Time (JIT) inventory policies often face a high risk of stockouts. Initially, this project aimed to predict stockouts within a 14-day window. However, an exploratory data audit revealed that the company's low-inventory baseline caused **94% of the catalog to be flagged as 'at risk'**, generating severe alert fatigue.

**The Solution:**
I engineered an end-to-end **72-hour Critical Alert System**. By narrowing the predictive window and preventing data leakage, the system reduces alert volume by 99% while isolating the highest-value revenue risks via a Telegram Bot simulation.

## 📊 Business Impact Score Formula

To support business prioritization and avoid alerting on low-value items, we compute:

`Business Impact Score = Stockout Probability × Relative Price × Sales Velocity`

Where:
- **Stockout Probability** → Model-predicted likelihood of depletion within 3 days.
- **Relative Price** → Scaled economic value of the product.
- **Sales Velocity** → Historical units sold (demand speed).

This allows ranking products not only by the probability of shortage but by **financial impact**, maximizing revenue protection.

## 🧠 Model Overview & MLOps
- **Algorithm:** XGBoost Classifier (optimized for tabular retail data).
- **Evaluation Metric:** ROC-AUC (**0.994** achieved on test set).
- **Key Methodological Decisions:**
  - **Data Leakage Prevention:** Removed deterministic variables (future demand forecast) from the feature space, forcing the model to learn genuine market patterns based on historical sales velocity, pricing, competitor aggressiveness, and temporal factors.
  - **Cost-Sensitive Learning:** Intentionally bypassed synthetic data generation (SMOTE) to preserve the real-world integrity of predicted probabilities. Class imbalance (69/31) was handled natively using `scale_pos_weight`.

## 📁 Repository Structure

📦 `retail-stockout-risk-scoring`
- `02_Data/`
  - `01_Raw/` - Original inventory dataset
- `03_Notebooks/`
  - `01_setup.ipynb` - Environment setup & library preparation
  - `02_data_quality.ipynb` - Data validation & cleaning checks
  - `03_eda.ipynb` - Exploratory Data Analysis & business logic validation
  - `04_feature_engineering.ipynb` - Target re-definition (72h) & cyclic time features
  - `05_feature_preselection.ipynb` - Data Leakage prevention & feature importance ranking
  - `06_modeling_classification.ipynb` - XGBoost/LightGBM training & probability thresholding
  - `07_production_framework_mlops.ipynb` - End-to-end MLOps scripts (Retraining, Inference & Alerts)
- `04_Models/` - Model artifacts
  - `champion_xgb_production.pkl` - Serialized pipeline loaded by Streamlit
- `app.py` - Streamlit scoring application
- `requirements.txt` - Python dependencies
- `README.md` - Documentation (this file)

## 📦 Installation (Local)

```bash
# Clone the repository
git clone [https://github.com/yourusername/retail-stockout-risk-scoring.git](https://github.com/yourusername/retail-stockout-risk-scoring.git)
cd retail-stockout-risk-scoring

# Create and activate environment
conda create -n stockout python=3.10
conda activate stockout

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py