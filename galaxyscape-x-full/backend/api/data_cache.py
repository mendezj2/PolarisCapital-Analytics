"""
Data Cache Module - Stores processed data for dashboard endpoints
"""
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
import json

# In-memory cache for processed datasets
_data_cache: Dict[str, pd.DataFrame] = {}
_file_timestamps: Dict[str, float] = {}


def get_cached_data(filepath: str) -> Optional[pd.DataFrame]:
    """Get cached data if file hasn't changed."""
    if not os.path.exists(filepath):
        return None
    
    current_mtime = os.path.getmtime(filepath)
    
    if filepath in _data_cache and _file_timestamps.get(filepath) == current_mtime:
        return _data_cache[filepath]
    
    return None


def cache_data(filepath: str, df: pd.DataFrame):
    """Cache processed data."""
    if os.path.exists(filepath):
        _data_cache[filepath] = df
        _file_timestamps[filepath] = os.path.getmtime(filepath)


def load_data_for_endpoint(filepath: str) -> Optional[pd.DataFrame]:
    """Load and cache data for API endpoints."""
    if not filepath or not os.path.exists(filepath):
        return None
    
    # Check cache first
    cached = get_cached_data(filepath)
    if cached is not None:
        return cached
    
    # Load fresh
    try:
        df = pd.read_csv(filepath, nrows=10000, low_memory=False)
        cache_data(filepath, df)
        return df
    except Exception:
        return None


def clear_cache():
    """Clear all cached data."""
    global _data_cache, _file_timestamps
    _data_cache.clear()
    _file_timestamps.clear()

