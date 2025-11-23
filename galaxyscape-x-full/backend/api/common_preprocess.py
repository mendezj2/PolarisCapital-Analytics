"""Shared preprocessing utilities."""
import pandas as pd
import numpy as np

def infer_schema(df):
    """Infer schema from dataframe."""
    schema = {
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'row_count': len(df),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_columns': detect_numeric_columns(df)
    }
    return schema

def detect_numeric_columns(df, threshold=0.9):
    """Detect numeric columns."""
    numeric_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            try:
                numeric_ratio = pd.to_numeric(df[col], errors='coerce').notna().sum() / len(df)
                if numeric_ratio >= threshold:
                    numeric_cols.append(col)
            except:
                pass
    return numeric_cols

def detect_domain_from_columns(columns):
    """Detect domain from column names."""
    lowered = {col.lower() for col in columns}
    astro_markers = {'luminosity', 'temperature', 'stellar_age', 'ra', 'dec', 'metallicity', 'magnitude'}
    finance_markers = {'ticker', 'close', 'volume', 'beta', 'portfolio', 'price', 'return', 'volatility'}
    
    astro_score = len(lowered & astro_markers)
    finance_score = len(lowered & finance_markers)
    
    if astro_score > finance_score:
        return 'astronomy'
    elif finance_score > astro_score:
        return 'finance'
    return 'unknown'

