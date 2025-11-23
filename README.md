# PolarisCapital Analytics

PolarisCapital Analytics is a dual-domain analytics platform that applies the same machine learning and data engineering engine to two very different worlds: **astronomy** and **finance**. It combines classical ML models, clean ETL patterns, and interactive dashboards to create a personal analytics lab for exploration and learning.

## Features

### Astronomy

- Star age prediction using gradient-boosted models (XGBoost / LightGBM)
- Clustering and dimensionality reduction (k-means, PCA / UMAP)
- Anomaly detection to flag unusual stars
- Sky map and network-style visualizations
- Real data from astronomy catalogs (e.g., exoplanet / stellar data)

### Finance

- Portfolio risk and volatility analytics
- Correlation networks across assets or sectors
- Monte Carlo-style outcome simulations
- Simple game-theory inspired views (Nash / Shapley style)
- Real data from finance APIs (e.g., yfinance)

### Platform

- Multiple interactive dashboards (astronomy + finance)
- Reusable frontend components (KPI cards, charts, tables, networks)
- CSV upload with automatic schema inference
- Architecture designed to be extended and used as a learning tool

## Tech Stack

- **Languages:** Python, JavaScript, SQL  
- **Backend:** Flask, pandas, numpy, scikit-learn, XGBoost, LightGBM  
- **Frontend:** HTML, CSS, vanilla JS, ECharts (plus some D3-style graphing)  
- **Data & Pipelines:** Kafka-style streaming, Snowflake-style schemas, Airflow / Prefect hooks  
- **Data Sources:** yfinance, astronomy catalog APIs (e.g., exoplanet / stellar archives)

## Getting Started

```bash
# Clone the repo
git clone https://github.com/<your-username>/polariscapital-analytics.git
cd polariscapital-analytics

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the backend
python run.py

# Then open frontend/static/index.html in your browser
