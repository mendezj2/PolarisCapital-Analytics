# GalaxyScape X - Multi-Domain Analytics Platform

A comprehensive analytics platform supporting both **Astronomy** and **Finance** domains with advanced ML capabilities, real-time streaming, and professional dashboards.

## 🏗️ Project Structure

The project is organized with clear **backend** and **frontend** separation:

- **`backend/`** - All Python code (API, ML models, data processing)
- **`frontend/`** - All static files (HTML, CSS, JavaScript)

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python run.py
# or
./start.sh
```

### 3. Access the Application
- **Main App**: http://localhost:5001
- **Health Check**: http://localhost:5001/health

## 📁 Key Directories

### Backend (`backend/`)
- **`api/`** - Flask REST API endpoints
- **`ml/`** - Machine learning models (XGBoost, LightGBM, PyTorch, etc.)
- **`data_eng/`** - Data engineering (Snowflake, Kafka streaming)
- **`data_sources/`** - Real dataset download modules
- **`config/`** - Configuration files

### Frontend (`frontend/`)
- **`static/`** - HTML, CSS, JavaScript files
  - **`css/`** - Stylesheets and themes
  - **`js/`** - JavaScript application code
    - **`components/`** - Reusable UI components

## ✨ Features

### Astronomy Mode
- Real dataset downloads (NASA Exoplanet Archive, Gaia DR3)
- Star age prediction (XGBoost, LightGBM)
- Clustering analysis (K-means, DBSCAN, HDBSCAN)
- Anomaly detection (IsolationForest, LOF)
- Network graph visualizations
- Cosmic Twin Finder

### Finance Mode
- Real market data downloads (Yahoo Finance, FRED)
- Risk scoring and volatility forecasting
- Correlation network analysis
- Real-time streaming analytics (Kafka)
- Portfolio analysis
- Anomaly detection

### Shared Features
- Professional Power BI-style dashboards
- Interactive visualizations (ECharts, D3.js)
- Data upload and processing
- ML model training and inference
- SHAP model explanations
- Network science utilities

## 🔧 Development

### Backend Development
All Python code is in `backend/`:
```python
# Import patterns within backend
from api.xxx import yyy
from ml.xxx import yyy
from data_sources.xxx import yyy
```

### Frontend Development
All static files are in `frontend/static/`:
- HTML: `frontend/static/index.html`
- CSS: `frontend/static/css/`
- JavaScript: `frontend/static/js/`

## 📊 API Endpoints

### Astronomy
- `POST /api/astronomy/upload` - Upload CSV
- `POST /api/astronomy/data/download` - Download real datasets
- `GET /api/astronomy/dashboard/kpi` - Get KPI metrics
- `POST /api/astronomy/predict` - Predict stellar ages
- `POST /api/astronomy/cosmic_twin` - Find similar stars

### Finance
- `POST /api/finance/upload` - Upload CSV
- `POST /api/finance/data/download` - Download real datasets
- `GET /api/finance/stream/risk` - Get real-time risk scores
- `POST /api/finance/predict` - Predict risk scores
- `GET /api/finance/stream/graph` - Get correlation network

## 📦 Dependencies

Key Python packages:
- Flask - Web framework
- pandas, numpy - Data processing
- scikit-learn - ML utilities
- xgboost, lightgbm - Gradient boosting
- torch - Deep learning
- shap - Model explainability
- networkx - Graph processing
- yfinance - Finance data

See `requirements.txt` for complete list.

## 🗂️ Data Storage

- **`data/raw/`** - Raw downloaded datasets
- **`data/processed/`** - Cleaned/processed data
- **`uploads/`** - User uploaded files

All data directories are gitignored.

## 📝 Documentation

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed structure
- [COMPLETION_STATUS.md](COMPLETION_STATUS.md) - Implementation status
- `dashboards/` - Dashboard specifications

## 🔐 Configuration

- API keys: Use environment variables (never hardcode)
- Data sources: Configure in `backend/data_sources/data_sources_config.json`
- Domain configs: `backend/config/`

## 🐳 Docker

Docker configuration available in `docker/` directory.

## 📄 License

See project license file.

## 🤝 Contributing

1. Backend code goes in `backend/`
2. Frontend code goes in `frontend/`
3. Follow existing patterns and structure
4. Test your changes before committing

---

**Built with**: Python, Flask, React-like components, ECharts, D3.js
