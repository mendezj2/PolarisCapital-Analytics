"""
Data Sources Module

This module provides functions to download real public datasets from online sources.
All implementations are skeleton/TODO style - you must complete the actual download logic.

Key Principles:
- Only real public datasets (no fake/toy data)
- No hardcoded API keys or secrets
- User must trigger downloads explicitly
- All data stored in data/raw/ before processing
"""

from .astronomy_download import (
    list_astronomy_sources,
    download_astronomy_sample,
    load_local_astronomy_raw
)

from .finance_download import (
    list_finance_sources,
    download_finance_sample,
    load_local_finance_raw
)

__all__ = [
    'list_astronomy_sources',
    'download_astronomy_sample',
    'load_local_astronomy_raw',
    'list_finance_sources',
    'download_finance_sample',
    'load_local_finance_raw'
]

