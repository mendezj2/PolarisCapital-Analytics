# GalaxyScape X - Project Structure

## Directory Organization

The project is now organized into clear **backend** and **frontend** separation:

```
galaxyscape-x-full/
├── backend/              # All Python backend code
│   ├── api/             # Flask API endpoints
│   │   ├── app.py       # Main Flask application
│   │   ├── astronomy_api.py
│   │   ├── finance_api.py
│   │   ├── common_preprocess.py
│   │   └── data_cache.py
│   ├── ml/              # Machine Learning models
│   │   ├── astronomy/   # Astronomy ML models
│   │   └── finance/     # Finance ML models
│   ├── data_eng/        # Data engineering
│   │   ├── ingestion/   # Data ingestion scripts
│   │   ├── streaming/   # Kafka streaming
│   │   └── snowflake_schema/
│   ├── data_sources/    # Data download modules
│   └── config/          # Configuration files
│
├── frontend/            # All frontend code
│   └── static/         # Static files (HTML, CSS, JS)
│       ├── index.html
│       ├── css/        # Stylesheets
│       └── js/         # JavaScript files
│           ├── components/  # UI components
│           ├── main.js
│           └── layout_manager.js
│
├── data/                # Data storage (gitignored)
│   ├── raw/            # Raw downloaded data
│   └── processed/      # Processed data
│
├── uploads/             # User uploaded files (gitignored)
│   ├── astronomy/
│   └── finance/
│
├── dashboards/         # Dashboard specifications (markdown)
├── docker/             # Docker configuration
├── requirements.txt    # Python dependencies
├── run.py             # Application entry point
└── start.sh           # Startup script
```

## Backend Structure

### API (`backend/api/`)
- **app.py**: Main Flask application, configures static files from `frontend/static`
- **astronomy_api.py**: Astronomy domain API endpoints
- **finance_api.py**: Finance domain API endpoints
- **common_preprocess.py**: Shared preprocessing utilities
- **data_cache.py**: Data caching for dashboard endpoints

### ML Models (`backend/ml/`)
- **astronomy/**: XGBoost, LightGBM, Autoencoder, Clustering, Anomaly Detection
- **finance/**: Risk models, LSTM, Correlation, Anomaly Detection
- **network_utils.py**: Network science utilities

### Data Engineering (`backend/data_eng/`)
- **ingestion/**: Snowflake ingestion scripts
- **streaming/**: Kafka producer/consumer
- **snowflake_schema/**: SQL schema definitions

### Data Sources (`backend/data_sources/`)
- **astronomy_download.py**: Download real astronomy datasets
- **finance_download.py**: Download real finance datasets

## Frontend Structure

### Static Files (`frontend/static/`)
- **index.html**: Main application page
- **css/**: Theme and component stylesheets
- **js/**: JavaScript application code
  - **components/**: Reusable UI components (charts, tables, etc.)
  - **main.js**: Application initialization
  - **layout_manager.js**: Dashboard layout management

## Running the Application

### Start the server:
```bash
python run.py
# or
./start.sh
```

The Flask app will:
1. Load from `backend/api/app.py`
2. Serve static files from `frontend/static/`
3. Run on http://localhost:5001

### Import Paths

**Within backend modules:**
- `from api.xxx import yyy` - Works within backend/
- `from ml.xxx import yyy` - Works within backend/
- `from data_sources.xxx import yyy` - Works within backend/

**From root (run.py):**
- Adds `backend/` to Python path
- Imports: `from api.app import app`

## Development Guidelines

### Backend Development
- All Python code goes in `backend/`
- Use relative imports within backend modules
- Add new API endpoints in `backend/api/`
- Add new ML models in `backend/ml/`

### Frontend Development
- All HTML/CSS/JS goes in `frontend/static/`
- Update `frontend/static/index.html` for new pages
- Add new components in `frontend/static/js/components/`
- CSS files in `frontend/static/css/`

### Data Files
- Raw data: `data/raw/` (gitignored)
- Processed data: `data/processed/` (gitignored)
- User uploads: `uploads/` (gitignored)

## Benefits of This Structure

1. **Clear Separation**: Backend and frontend are completely separate
2. **Easy Deployment**: Can deploy backend and frontend independently
3. **Better Organization**: Related files are grouped together
4. **Scalability**: Easy to add new backend services or frontend frameworks
5. **Team Collaboration**: Frontend and backend developers can work independently




