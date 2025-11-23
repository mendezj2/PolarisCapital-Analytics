"""
File Management System
Manages CSV files per dashboard, ensuring each dashboard uses its own data files.
"""
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_DATASETS = {
    'astronomy': PROJECT_ROOT / 'data' / 'raw' / 'astronomy' / 'default_astronomy_dataset.csv',
    'finance': PROJECT_ROOT / 'data' / 'raw' / 'finance' / 'default_finance_dataset.csv'
}

# Dashboard to file mappings
DASHBOARD_FILE_MAPPINGS = {
    'astronomy': {
        'overview': ['star_explorer.csv', 'nasa_exoplanets.csv'],
        'star-explorer': ['star_explorer.csv'],
        'star-age': ['star_explorer.csv', 'ml_models.csv'],
        'sky-map': ['sky_map.csv', 'star_explorer.csv'],
        'light-curve': ['star_explorer.csv'],
        'clusters': ['cluster_analysis.csv', 'star_explorer.csv'],
        'anomalies': ['anomaly_detection.csv', 'star_explorer.csv'],
        'sky-network': ['sky_map.csv'],
        'ml-models': ['ml_models.csv', 'star_explorer.csv']
    },
    'finance': {
        'risk': ['risk_dashboard.csv', 'market_data.csv'],
        'streaming': ['market_data.csv', 'risk_dashboard.csv'],
        'correlation': ['correlation_network.csv', 'market_data.csv'],
        'portfolio': ['market_data.csv'],
        'compliance': ['risk_dashboard.csv'],
        'stock-explorer': ['stock_explorer.csv', 'market_data.csv'],
        'future-outcomes': ['future_outcomes.csv', 'market_data.csv'],
        'marketing-analytics': ['marketing_signage.csv', 'marketing_omni_channel.csv'],
        'ml-models': ['market_data.csv'],
        'game-theory': ['game_theory.csv', 'market_data.csv']
    }
}

# Active file selections per dashboard (can be updated via API)
_active_files: Dict[str, Dict[str, str]] = {
    'astronomy': {},
    'finance': {}
}


def get_base_data_dir(domain: str) -> Path:
    """Get base data directory for domain."""
    base_dir = Path(__file__).parent.parent.parent
    return base_dir / 'data' / 'raw' / domain


def list_available_files(domain: str, dashboard: Optional[str] = None) -> List[Dict]:
    """
    List available CSV files for a domain/dashboard.
    
    Returns:
        List of file info dicts with name, path, row_count, columns, size
    """
    base_dir = get_base_data_dir(domain)
    uploads_dir = base_dir.parent.parent / 'uploads' / domain
    
    files = []
    
    # Check raw data directory
    if base_dir.exists():
        for csv_file in sorted(base_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file, nrows=1)
                file_type = 'default' if csv_file == DEFAULT_DATASETS.get(domain) else 'raw'
                file_info = {
                    'name': csv_file.name,
                    'path': str(csv_file),
                    'type': file_type,
                    'row_count': _count_rows(csv_file),
                    'columns': df.columns.tolist(),
                    'size': csv_file.stat().st_size,
                    'dashboard': _get_dashboard_for_file(domain, csv_file.name)
                }
                if dashboard is None or csv_file.name in DASHBOARD_FILE_MAPPINGS.get(domain, {}).get(dashboard, []):
                    files.append(file_info)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    # Check uploads directory
    if uploads_dir.exists():
        for csv_file in sorted(uploads_dir.glob('*.csv')):
            try:
                df = pd.read_csv(csv_file, nrows=1)
                file_info = {
                    'name': csv_file.name,
                    'path': str(csv_file),
                    'type': 'uploaded',
                    'row_count': _count_rows(csv_file),
                    'columns': df.columns.tolist(),
                    'size': csv_file.stat().st_size,
                    'dashboard': _get_dashboard_for_file(domain, csv_file.name)
                }
                if dashboard is None or True:  # Uploads available to all
                    files.append(file_info)
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    
    return files


def _count_rows(filepath: Path) -> int:
    """Count rows in CSV file efficiently."""
    try:
        with open(filepath, 'r') as f:
            return sum(1 for _ in f) - 1  # Subtract header
    except:
        return 0


def _get_dashboard_for_file(domain: str, filename: str) -> List[str]:
    """Get dashboards that use this file."""
    dashboards = []
    for dash, files in DASHBOARD_FILE_MAPPINGS.get(domain, {}).items():
        if filename in files:
            dashboards.append(dash)
    return dashboards


def get_default_dataset_path(domain: str) -> Optional[str]:
    path = DEFAULT_DATASETS.get(domain)
    if path and Path(path).exists():
        return str(path)
    return None


def get_active_file(domain: str, dashboard: str) -> Optional[str]:
    """Get the currently active file for a dashboard."""
    # Check if user has selected a specific file
    if domain in _active_files and dashboard in _active_files[domain]:
        selected = _active_files[domain][dashboard]
        if os.path.exists(selected):
            return selected
    
    # Use domain default dataset first
    default_path = get_default_dataset_path(domain)
    if default_path:
        return default_path
    
    # Use legacy mapping
    base_dir = get_base_data_dir(domain)
    default_files = DASHBOARD_FILE_MAPPINGS.get(domain, {}).get(dashboard, [])
    for filename in default_files:
        filepath = base_dir / filename
        if filepath.exists():
            return str(filepath)
    
    # Fallback: try uploads
    uploads_dir = base_dir.parent.parent / 'uploads' / domain
    if uploads_dir.exists():
        uploaded_files = sorted(uploads_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if uploaded_files:
            return str(uploaded_files[0])
    
    return None


def set_active_file(domain: str, dashboard: str, filepath: str):
    """Set the active file for a dashboard."""
    if domain not in _active_files:
        _active_files[domain] = {}
    _active_files[domain][dashboard] = filepath


def set_active_file_for_domain(domain: str, filepath: str):
    """Point every dashboard in a domain to the same file (used after uploads)."""
    if domain not in _active_files:
        _active_files[domain] = {}
    for dash in DASHBOARD_FILE_MAPPINGS.get(domain, {}):
        _active_files[domain][dash] = filepath


def reset_to_default(domain: str):
    """Clear custom selections so the domain falls back to the default dataset."""
    _active_files[domain] = {}


def load_dashboard_data(domain: str, dashboard: str) -> Optional[pd.DataFrame]:
    """
    Load data for a specific dashboard using its active file.
    
    Returns:
        DataFrame or None if no file found
    """
    filepath = get_active_file(domain, dashboard)
    if not filepath or not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        return df
    except Exception as e:
        print(f"Error loading {filepath} for {dashboard}: {e}")
        return None


def get_file_info(domain: str, dashboard: str) -> Dict:
    """Get information about the active file for a dashboard."""
    filepath = get_active_file(domain, dashboard)
    if not filepath:
        return {
            'active_file': None,
            'available_files': list_available_files(domain, dashboard),
            'dashboard': dashboard
        }
    
    try:
        df = pd.read_csv(filepath, nrows=1)
        return {
            'active_file': {
                'name': os.path.basename(filepath),
                'path': filepath,
                'row_count': _count_rows(Path(filepath)),
                'columns': df.columns.tolist(),
                'size': os.path.getsize(filepath)
            },
            'available_files': list_available_files(domain, dashboard),
            'dashboard': dashboard
        }
    except Exception as e:
        return {
            'active_file': None,
            'available_files': list_available_files(domain, dashboard),
            'dashboard': dashboard,
            'error': str(e)
        }

