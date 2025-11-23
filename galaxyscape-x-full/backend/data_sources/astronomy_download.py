"""
Astronomy Data Download - Complete Implementation

Downloads real astronomy datasets from public sources.
"""
from __future__ import annotations

import os
import json
import requests
from pathlib import Path
from typing import Any
import pandas as pd
from urllib.parse import urlparse


def list_astronomy_sources() -> list[dict[str, Any]]:
    """Return a list of known public astronomy data sources."""
    config_path = Path(__file__).parent / 'data_sources_config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            return list(config.get('astronomy', {}).get('sources', {}).values())
    
    sources = [
        {
            'name': 'NASA Exoplanet Archive',
            'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&format=csv',
            'description': 'Confirmed exoplanets with orbital and physical parameters',
            'format': 'CSV',
            'size_mb': 5,
            'columns': ['pl_name', 'hostname', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'st_teff', 'st_rad', 'st_mass']
        },
        {
            'name': 'Gaia DR3 Sample',
            'url': 'https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_tables/ssec_dm_gaia_source.html',
            'description': 'Gaia Data Release 3 stellar catalog (requires manual export)',
            'format': 'CSV',
            'size_mb': 50,
            'note': 'Manual download required from Gaia archive portal'
        }
    ]
    return sources


def download_astronomy_sample(output_path: str, source_name: str | None = None) -> str:
    """
    Download a real public astronomy dataset to data/raw/astronomy.
    
    Args:
        output_path: Full path where downloaded file should be saved.
        source_name: Optional name of source from list_astronomy_sources().
    
    Returns:
        Path to downloaded file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sources = list_astronomy_sources()
    source = None
    
    if source_name:
        source = next((s for s in sources if s.get('name') == source_name), None)
    
    if not source:
        # Default to NASA Exoplanet Archive
        source = next((s for s in sources if 'exoplanet' in s.get('name', '').lower()), sources[0] if sources else None)
    
    if not source or not source.get('url'):
        raise ValueError(f"Source not found or no URL available: {source_name}")
    
    url = source['url']
    
    # Handle NASA Exoplanet Archive
    if 'exoplanetarchive.ipac.caltech.edu' in url:
        try:
            response = requests.get(url, timeout=300, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
        except Exception as e:
            raise Exception(f"Failed to download from {url}: {str(e)}")
    
    # Generic download
    try:
        response = requests.get(url, timeout=300, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download: {str(e)}")


def load_local_astronomy_raw(file_path: str | None = None) -> pd.DataFrame:
    """
    Load a previously-downloaded astronomy file from data/raw/astronomy.
    
    Args:
        file_path: Optional specific file path. If None, loads most recent file.
    
    Returns:
        pandas DataFrame with astronomy data.
    """
    raw_dir = Path('data/raw/astronomy')
    
    if file_path is None:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
        files = sorted(raw_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not files:
            raise FileNotFoundError(f"No CSV files found in {raw_dir}")
        file_path = str(files[0])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Try to detect encoding and handle various formats
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    df = None
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise Exception(f"Failed to read {file_path}: {str(e)}")
    
    if df is None:
        raise Exception(f"Could not read {file_path} with any encoding")
    
    return df


def validate_astronomy_data(df: pd.DataFrame) -> dict[str, Any]:
    """
    Validate downloaded astronomy data for required columns and data quality.
    
    Args:
        df: DataFrame to validate.
    
    Returns:
        Dictionary with validation results.
    """
    issues = []
    
    # Common astronomy columns (flexible validation)
    common_cols = ['ra', 'dec', 'magnitude', 'temperature', 'luminosity', 'mass']
    found_cols = [col for col in common_cols if col in df.columns]
    
    if len(found_cols) == 0:
        # Check for alternative column names
        alt_cols = ['pl_name', 'hostname', 'st_teff', 'st_rad', 'st_mass']  # Exoplanet archive
        found_cols = [col for col in alt_cols if col in df.columns]
        if len(found_cols) == 0:
            issues.append('No recognized astronomy columns found')
    
    # Check for reasonable data ranges
    if 'ra' in df.columns:
        invalid_ra = df[(df['ra'] < 0) | (df['ra'] > 360)]
        if len(invalid_ra) > 0:
            issues.append(f"Found {len(invalid_ra)} rows with invalid RA (should be 0-360)")
    
    if 'dec' in df.columns:
        invalid_dec = df[(df['dec'] < -90) | (df['dec'] > 90)]
        if len(invalid_dec) > 0:
            issues.append(f"Found {len(invalid_dec)} rows with invalid Dec (should be -90 to 90)")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'row_count': len(df),
        'column_count': len(df.columns),
        'columns': list(df.columns)
    }
