# Data Sources Module - Real Public Datasets Only

## Purpose

This module provides skeleton functions to download **REAL public datasets** from online sources. 
**No fake or toy CSVs are included** - you must either:
1. Upload your own real CSV files, or
2. Trigger downloads from actual public data sources

## Key Principles

- ✅ **Real datasets only** - No hardcoded sample/toy data
- ✅ **Public sources** - Only publicly available datasets
- ✅ **No secrets** - API keys stored in environment variables, never hardcoded
- ✅ **User-triggered** - Downloads only happen when you explicitly call functions
- ✅ **Learning-first** - All implementations are TODOs for you to complete

## Folder Structure

```
data/
├── raw/
│   ├── astronomy/     # Downloaded raw astronomy datasets
│   └── finance/        # Downloaded raw finance datasets
└── processed/
    ├── astronomy/      # Cleaned/processed astronomy data
    └── finance/        # Cleaned/processed finance data
```

## Usage

### 1. List Available Sources

```python
from data_sources import list_astronomy_sources, list_finance_sources

# See available astronomy sources
astronomy_sources = list_astronomy_sources()
print(astronomy_sources)

# See available finance sources
finance_sources = list_finance_sources()
print(finance_sources)
```

### 2. Download Real Data

```python
from data_sources import download_astronomy_sample, download_finance_sample

# Download astronomy data (TODO: implement actual download)
download_astronomy_sample('data/raw/astronomy/gaia_sample.csv')

# Download finance data (TODO: implement actual download)
download_finance_sample('data/raw/finance/sp500_sample.csv', tickers=['AAPL', 'MSFT'])
```

### 3. Load Downloaded Data

```python
from data_sources import load_local_astronomy_raw, load_local_finance_raw

# Load most recent astronomy file
df_astro = load_local_astronomy_raw()

# Load specific finance file
df_finance = load_local_finance_raw('data/raw/finance/sp500_sample.csv')
```

## Configuration

Edit `data_sources_config.json` to:
- Replace `TODO_USER_*` placeholders with actual dataset URLs
- Configure default sources
- Set up API key environment variable names (never put keys in the file)

## Example Public Data Sources

### Astronomy
- **Gaia DR3**: https://gea.esac.esa.int/archive/ (stellar catalog)
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/ (exoplanet data)
- **Kepler/TESS**: https://archive.stsci.edu/ (light curves)
- **SDSS**: https://www.sdss.org/dr17/ (sky survey data)

### Finance
- **Yahoo Finance**: Historical stock prices (use yfinance library or CSV exports)
- **FRED**: https://fred.stlouisfed.org/ (economic indicators)
- **Alpha Vantage**: https://www.alphavantage.co/ (free stock API, requires API key)
- **Kaggle**: https://www.kaggle.com/datasets (public financial datasets)

## API Keys and Authentication

**Never hardcode API keys or credentials in code.**

Instead:
1. Get API key from the data provider (if required)
2. Set as environment variable:
   ```bash
   export ALPHA_VANTAGE_API_KEY="your_key_here"
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_key"
   ```
3. Access in code via `os.getenv('ALPHA_VANTAGE_API_KEY')`

## Implementation TODOs

All functions in this module are skeletons. You must implement:

1. **Download functions**: Actual HTTP requests or library calls
2. **File loading**: Format detection and appropriate readers
3. **Validation**: Data quality checks
4. **Error handling**: Network errors, file errors, authentication failures
5. **Progress tracking**: For large downloads

## Data License Compliance

**You are responsible for:**
- Complying with dataset licenses and terms of service
- Respecting rate limits and usage restrictions
- Properly attributing data sources
- Not redistributing proprietary datasets

## Integration with Pipeline

Once downloaded, data flows through:
1. `data/raw/` → Raw downloaded files
2. `ml/*/preprocess.py` → Cleaning and normalization
3. `data/processed/` → Cleaned data ready for ML
4. ML models → Training and inference
5. Dashboards → Visualization

See `ml/astronomy/preprocess.py` and `ml/finance/preprocess.py` for `load_and_clean_from_raw()` functions.

## Troubleshooting

- **Download fails**: Check URL, network connection, API key (if required)
- **File not found**: Ensure download completed successfully
- **Format errors**: Check file format (CSV, TSV, FITS) and use appropriate reader
- **Large files**: Consider chunking or streaming for very large datasets

