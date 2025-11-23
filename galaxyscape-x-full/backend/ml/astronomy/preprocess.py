"""Astronomy-specific preprocessing."""
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_and_clean_from_raw(file_path: str | None = None, auto_download: bool = False) -> pd.DataFrame:
    """
    High-level convenience function: load raw data, clean it, return ready-for-ML DataFrame.

    Args:
        file_path: Optional specific file path. If None, loads most recent file from data/raw/astronomy.
        auto_download: If True and no local file found, attempt to download sample data.

    Returns:
        Cleaned pandas DataFrame.
    """
    import sys
    import os
    backend_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, backend_path)
    
    from data_sources.astronomy_download import (
        load_local_astronomy_raw, 
        download_astronomy_sample, 
        validate_astronomy_data
    )
    
    raw_dir = Path('data/raw/astronomy')
    processed_dir = Path('data/processed/astronomy')
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
            # Download sample
            output_path = str(raw_dir / 'astronomy_sample.csv')
            file_path = download_astronomy_sample(output_path)
        else:
            raise FileNotFoundError('No raw data file found. Set auto_download=True or download manually.')
    
    # Load raw data
    df_raw = load_local_astronomy_raw(file_path)
    
    # Clean
    df_cleaned = clean_astronomy_df(df_raw)
    
    # Validate
    validation = validate_astronomy_data(df_cleaned)
    if not validation['valid']:
        print(f"Validation warnings: {validation['issues']}")
    
    # Save processed version
    processed_path = processed_dir / Path(file_path).name
    df_cleaned.to_csv(processed_path, index=False)
    
    return df_cleaned


def clean_astronomy_df(df):
    """Clean astronomy dataframe."""
    df = df.copy()
    # Remove negative luminosities
    if 'luminosity' in df.columns:
        df = df[df['luminosity'] > 0]
    # Remove invalid temperatures
    if 'temperature' in df.columns:
        df = df[(df['temperature'] > 0) & (df['temperature'] < 100000)]
    return df

def normalize_features(df, feature_cols):
    """Normalize features using z-score."""
    normalized = df.copy()
    for col in feature_cols:
        if col in normalized.columns:
            mean = normalized[col].mean()
            std = normalized[col].std()
            if std > 0:
                normalized[col] = (normalized[col] - mean) / std
    return normalized

def create_embedding_inputs(df):
    """Prepare inputs for embedding models."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = df[numeric_cols].fillna(0).values
    return {
        'features': features.tolist(),
        'feature_names': numeric_cols,
        'metadata': df[['id'] if 'id' in df.columns else []].to_dict('records') if 'id' in df.columns else []
    }

