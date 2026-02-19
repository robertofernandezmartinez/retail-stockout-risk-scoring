# 🛒 Retail Stockout Risk Scoring & MLOps Pipeline

This project predicts **proactive stockout risks within a 14-day international replenishment window** and estimates the **economic impact** of potential shortages. 

By applying a custom Business Logic layer to a calibrated XGBoost model, the system identifies high-value replenishment risks, allowing supply chain managers to act before the stock hits zero.

## 🚀 Live Application

Try the interactive "What-if" simulation dashboard here 👇
👉 **https://retail-stockout-risk-scoring.streamlit.app/**

## 📌 Executive Summary
Retailers operating with global supply chains often face a high risk of stockouts. Initially, this project aimed to predict stockouts using deterministic rules (Inventory < 10), but an exploratory data audit revealed a **Data Leakage trap** that caused a false 1.0 AUC, generating severe alert fatigue.

**The Solution:**
I engineered an end-to-end **Strategic 14-day Warning System**. By injecting **5% stochastic noise** and re-engineering the target signal based on **Sales Velocity vs. Lead-Time**, the model achieved a robust and realistic **AUC of 0.91**. This system isolates high-value revenue risks via a daily MLOps inference suite.

## 📊 Business Impact Score Formula

To support business prioritization and avoid alerting on low-value items, we compute:

`Business Impact Score = Stockout Probability × Price × Sales Velocity`

Where:
- **Stockout Probability** → Calibrated likelihood of depletion within the 14-day window.
- **Price** → Unit economic value of the product.
- **Sales Velocity** → Historical units sold (demand speed).

This allows ranking products not only by the probability of shortage but by **financial impact**, maximizing revenue protection for the company.

## 🧠 Model Overview & MLOps
- **Algorithm:** XGBoost Classifier (optimized for tabular retail data).
- **Evaluation Metric:** ROC-AUC (**0.9085** achieved on realistic, noisy data).
- **Key Methodological Decisions:**
  - **Stochastic Decoupling:** Injected Gaussian noise to simulate real-world ERP lags, forcing the model to learn genuine market patterns rather than hard-coded thresholds.
  - **Data Leakage Prevention:** Removed deterministic variables (future demand) from the feature space to ensure the model remains robust in production.
  - **Cost-Sensitive Learning:** Handled class imbalance (85/15) natively using `scale_pos_weight` to preserve the integrity of predicted probabilities for business decision-making.

## 📁 Repository Structure

📦 `retail-stockout-risk-scoring`
- `02_Data/`
  - `01_Raw/` - Original inventory dataset (`retail_store_inventory.csv`)
- `03_Notebooks/`
  - `01_setup_and_healing.ipynb` - Environment setup & Stochastic Noise Injection
  - `02_eda.ipynb` - Data validation & Leakage identification
  - `03_feature_engineering.ipynb` - 14-day Target re-definition & cyclic time features
  - `04_feature_preselection.ipynb` - Leakage prevention & feature importance ranking
  - `05_modeling_classification.ipynb` - XGBoost training & AUC validation (0.91)
  - `06_production_framework_mlops.ipynb` - End-to-end MLOps scripts (Retraining & Alerts)
  - `07_streamlit_pipeline_packaging.ipynb` - Serialization of the "Black-Box" Pipeline artifact
- `04_Models/`
  - `full_pipeline_14day_strategic.pkl` - Serialized pipeline loaded by Streamlit
- `app.py` - Streamlit simulation application
- `requirements.txt` - Python dependencies
- `README.md` - Documentation (this file)

## 📦 Installation (Local)

```bash
# Clone the repository
git clone [https://github.com/yourusername/retail-stockout-risk-scoring.git](https://github.com/yourusername/retail-stockout-risk-scoring.git)
cd retail-stockout-risk-scoring

# Create and activate environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py