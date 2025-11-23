"""Finance-specific preprocessing."""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_clean_from_raw(file_path: str | None = None, auto_download: bool = False, tickers: list[str] | None = None) -> pd.DataFrame:
    """
    High-level convenience function: load raw finance data, clean it, return ready-for-ML DataFrame.

    Args:
        file_path: Optional specific file path. If None, loads most recent file from data/raw/finance.
        auto_download: If True and no local file found, attempt to download sample data.
        tickers: Optional list of tickers to download.

    Returns:
        Cleaned pandas DataFrame.
    """
    import sys
    import os
    backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, backend_path)
    
    from data_sources.finance_download import (
        load_local_finance_raw, 
        download_finance_sample, 
        validate_finance_data
    )
    
    raw_dir = Path('data/raw/finance')
    processed_dir = Path('data/processed/finance')
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing file or download
    if file_path is None:
        if raw_dir.exists():
            files = sorted(raw_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
            file_path = str(files[0]) if files else None
        else:
            file_path = None
    
    if file_path is None or not Path(file_path).exists():
        if auto_download:
            output_path = str(raw_dir / 'finance_sample.csv')
            file_path = download_finance_sample(output_path, tickers=tickers)
        else:
            raise FileNotFoundError('No raw data file found. Set auto_download=True or download manually.')
    
    # Load raw data
    df_raw = load_local_finance_raw(file_path)
    
    # Engineer features
    df_cleaned = engineer_risk_features(df_raw)
    
    # Validate
    validation = validate_finance_data(df_cleaned)
    if not validation['valid']:
        print(f"Validation warnings: {validation['issues']}")
    
    # Save processed version
    processed_path = processed_dir / Path(file_path).name
    df_cleaned.to_csv(processed_path, index=False)
    
    return df_cleaned


def build_timeseries_panels(df, entity_col='ticker'):
    """Build time series panels per entity."""
    panels = {}
    if entity_col in df.columns:
        for entity, group in df.groupby(entity_col):
            sorted_group = group.sort_values('date' if 'date' in group.columns else group.index)
            panels[entity] = sorted_group
    else:
        # Fallback: use index
        panels['default'] = df
    return panels

def engineer_risk_features(df):
    """Engineer risk features."""
    engineered = df.copy()
    
    # Find price column
    price_col = None
    for col in ['Close', 'close', 'CLOSE', 'Price', 'price', 'Adj Close', 'Adj_Close']:
        if col in engineered.columns:
            price_col = col
            break
    
    if price_col:
        # Calculate returns
        engineered['returns'] = engineered[price_col].pct_change()
        
        # Rolling volatility (annualized)
        engineered['volatility_30d'] = engineered['returns'].rolling(window=30, min_periods=1).std() * np.sqrt(252)
        engineered['volatility_7d'] = engineered['returns'].rolling(window=7, min_periods=1).std() * np.sqrt(252)
        engineered['volatility_90d'] = engineered['returns'].rolling(window=90, min_periods=1).std() * np.sqrt(252)
        
        # Moving averages
        engineered['ma_20'] = engineered[price_col].rolling(window=20, min_periods=1).mean()
        engineered['ma_50'] = engineered[price_col].rolling(window=50, min_periods=1).mean()
        
        # Price change
        engineered['price_change'] = engineered[price_col].diff()
        engineered['price_change_pct'] = engineered[price_col].pct_change() * 100
    
    # Volume features if available
    volume_col = None
    for col in ['Volume', 'volume', 'VOLUME']:
        if col in engineered.columns:
            volume_col = col
            break
    
    if volume_col:
        engineered['volume_ma_20'] = engineered[volume_col].rolling(window=20, min_periods=1).mean()
        engineered['volume_ratio'] = engineered[volume_col] / (engineered['volume_ma_20'] + 1e-6)
    
    # Fill NaN values
    engineered = engineered.fillna(0)
    
    return engineered

def validate_finance_schema(schema):
    """Validate finance schema."""
    issues = []
    
    required_cols = ['ticker', 'date', 'price']
    for col in required_cols:
        if col not in schema.get('columns', []):
            issues.append(f'Missing required column: {col}')
    
    schema['issues'] = issues
    schema['valid'] = len(issues) == 0
    return schema

