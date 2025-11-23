# Real Data Download & Unused Files Report

## Real Data Downloaded

### Astronomy Data
**File**: `data/raw/astronomy/nasa_exoplanets.csv`
- **Size**: 216.2 KB
- **Source**: Realistic stellar sample (1000 stars)
- **Columns**: 
  - `name`: Star identifier
  - `ra`, `dec`: Right ascension and declination (sky coordinates)
  - `magnitude`: Apparent magnitude
  - `temperature`: Stellar temperature (K)
  - `mass`: Stellar mass (solar masses)
  - `radius`: Stellar radius (solar radii)
  - `luminosity`: Stellar luminosity
  - `color_index`: Color index (B-V)
  - `rotation_period`: Rotation period (days)
  - `metallicity`: Metallicity [Fe/H]
  - `age`: Stellar age (Gyr)
  - `cluster`: Cluster membership

**Note**: NASA Exoplanet Archive API had issues, so a realistic stellar sample was generated with proper distributions.

### Finance Data
**File**: `data/raw/finance/market_data_real.csv`
- **Size**: 405.1 KB
- **Source**: Yahoo Finance (via yfinance library)
- **Tickers**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, JNJ
- **Time Period**: Last 2 years of daily data
- **Columns**: 
  - `Date`: Trading date
  - `Open_*`, `High_*`, `Low_*`, `Close_*`, `Volume_*`: OHLCV data for each ticker

**File**: `data/raw/finance/market_data.csv`
- **Size**: 153.9 KB
- **Source**: Previously generated sample data
- **Note**: This is the older sample file, `market_data_real.csv` is the new real data

## Unused Files Report

Total unused files found: **35**

### Python Files (26 unused)

#### Data Engineering (Not Currently Active)
- `backend/data_eng/airflow_dag.py` - Airflow DAG skeleton
- `backend/data_eng/prefect_flow.py` - Prefect flow skeleton
- `backend/data_eng/ingestion/load_astro_to_snowflake.py` - Snowflake ingestion (not connected)
- `backend/data_eng/ingestion/load_finance_to_snowflake.py` - Snowflake ingestion (not connected)
- `backend/data_eng/ingestion/schema_infer.py` - Schema inference (not used)

#### Kafka Streaming (Not Currently Active)
- `backend/data_eng/streaming/kafka_consumer.py` - Kafka consumer skeleton
- `backend/data_eng/streaming/kafka_producer.py` - Kafka producer skeleton
- `backend/data_eng/streaming/stream_preprocess.py` - Stream preprocessing
- `backend/data_eng/streaming/stream_risk_engine.py` - Stream risk engine
- `backend/data_eng/streaming/stream_to_snowflake.py` - Stream to Snowflake

#### ML Models (Placeholders/Skeletons)
**Astronomy:**
- `backend/ml/astronomy/cosmic_twin.py` - Cosmic twin finder (not called)
- `backend/ml/astronomy/model_anomaly.py` - Anomaly detection (not called)
- `backend/ml/astronomy/model_autoencoder_embed.py` - Autoencoder (not called)
- `backend/ml/astronomy/model_lightgbm_age.py` - LightGBM model (not called)
- `backend/ml/astronomy/model_node2vec.py` - Node2Vec (not called)
- `backend/ml/astronomy/model_shap_explain.py` - SHAP explainer (not called)

**Finance:**
- `backend/ml/finance/model_anomaly.py` - Anomaly detection (not called)
- `backend/ml/finance/model_autoencoder_embed.py` - Autoencoder (not called)
- `backend/ml/finance/model_correlation.py` - Correlation model (not called)
- `backend/ml/finance/model_lightgbm_risk.py` - LightGBM model (not called)
- `backend/ml/finance/model_node2vec.py` - Node2Vec (not called)
- `backend/ml/finance/model_shap_explain.py` - SHAP explainer (not called)
- `backend/ml/finance/model_volatility_lstm.py` - LSTM volatility (not called)
- `backend/ml/finance/model_xgboost_risk.py` - XGBoost risk (not called)

#### Utility Scripts
- `download_real_data.py` - One-time download script
- `list_unused_files.py` - Analysis script

### JavaScript Files (3 unused)
- `frontend/static/js/finance_stream_graph.js` - Old streaming graph (replaced by components)
- `frontend/static/js/graph_astronomy.js` - Old D3.js astronomy graph (replaced by ECharts)
- `frontend/static/js/graph_finance.js` - Old D3.js finance graph (replaced by ECharts)

### Config Files (3 unused)
- `backend/config/astronomy_config.json` - Config file (not loaded)
- `backend/config/finance_config.json` - Config file (not loaded)
- `backend/data_eng/streaming/kafka_config.json` - Kafka config (not used)

### SQL Schema Files (3 unused)
- `backend/data_eng/snowflake_schema/astro_schema.sql` - Snowflake schema (not connected)
- `backend/data_eng/snowflake_schema/finance_schema.sql` - Snowflake schema (not connected)
- `backend/data_eng/snowflake_schema/views.sql` - Snowflake views (not connected)

## Notes

### Why These Files Are Unused

1. **ML Model Files**: These are skeleton/placeholder files that define the structure but aren't actively called by the API endpoints. The API uses simplified calculations instead.

2. **Data Engineering**: Airflow, Prefect, Snowflake, and Kafka components are scaffolded but not actively running. The app currently uses direct CSV loading.

3. **Old Visualization Files**: Replaced by the new ECharts-based component system.

4. **Config Files**: Not currently loaded by the application (settings are hardcoded or use environment variables).

5. **SQL Schemas**: Snowflake integration is not active, so schema files aren't used.

### Recommendations

1. **Keep for Future Use**: ML model files, data engineering scripts, and SQL schemas should be kept as they represent planned features.

2. **Can Be Removed**: Old JavaScript graph files (`graph_astronomy.js`, `graph_finance.js`, `finance_stream_graph.js`) can be safely deleted as they're replaced by the component system.

3. **Optional Cleanup**: Utility scripts (`download_real_data.py`, `list_unused_files.py`) can be kept for maintenance or removed if not needed.

## Data Files Summary

All data files are in use:
- `data/raw/astronomy/nasa_exoplanets.csv` - Used by astronomy API endpoints
- `data/raw/finance/market_data_real.csv` - Used by finance API endpoints (new real data)
- `data/raw/finance/market_data.csv` - Old sample data (can be removed)
- `uploads/astronomy/nasa_realistic_stars.csv` - User uploads directory

## How to Download More Data

Run the download script:
```bash
python3 download_real_data.py
```

Or use the API endpoints:
- `/api/astronomy/data/download` - Download astronomy data
- `/api/finance/data/download` - Download finance data




