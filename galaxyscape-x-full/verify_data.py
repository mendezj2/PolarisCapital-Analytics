#!/usr/bin/env python3
"""
Quick verification script to check data files and API endpoints work.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add backend to path
project_root = Path(__file__).parent
backend_path = project_root / 'backend'
sys.path.insert(0, str(backend_path))

def check_data_files():
    """Check if data files exist and are readable."""
    print("="*60)
    print("Checking Data Files")
    print("="*60)
    
    files_to_check = {
        'Astronomy': [
            'data/raw/astronomy/nasa_exoplanets.csv',
            'data/raw/astronomy/nasa_realistic_stars.csv',
        ],
        'Finance': [
            'data/raw/finance/market_data.csv',
            'data/raw/finance/market_data_real.csv',
        ]
    }
    
    all_ok = True
    for domain, files in files_to_check.items():
        print(f"\n{domain}:")
        for filepath in files:
            full_path = project_root / filepath
            if full_path.exists():
                try:
                    df = pd.read_csv(full_path, nrows=5)
                    print(f"  ✅ {filepath}")
                    print(f"     Rows: {len(pd.read_csv(full_path))}, Columns: {list(df.columns)[:5]}...")
                except Exception as e:
                    print(f"  ❌ {filepath} - Error reading: {e}")
                    all_ok = False
            else:
                print(f"  ⚠️  {filepath} - Not found")
                if 'nasa_realistic_stars' in filepath:
                    print(f"     (Will use nasa_exoplanets.csv as fallback)")
    
    return all_ok

def test_data_loading():
    """Test data loading functions."""
    print("\n" + "="*60)
    print("Testing Data Loading Functions")
    print("="*60)
    
    try:
        from api.astronomy_api import _get_astronomy_data
        from api.finance_api import _get_finance_data
        
        # Test astronomy
        print("\nAstronomy Data:")
        astro_df = _get_astronomy_data()
        if astro_df is not None and len(astro_df) > 0:
            print(f"  ✅ Loaded {len(astro_df)} rows")
            print(f"     Columns: {list(astro_df.columns)[:8]}")
        else:
            print("  ❌ Failed to load astronomy data")
        
        # Test finance
        print("\nFinance Data:")
        finance_df = _get_finance_data()
        if finance_df is not None and len(finance_df) > 0:
            print(f"  ✅ Loaded {len(finance_df)} rows")
            print(f"     Columns: {list(finance_df.columns)[:8]}")
            print(f"     Date range: {finance_df['Date'].min()} to {finance_df['Date'].max()}")
            print(f"     Tickers: {finance_df['Ticker'].nunique()} unique")
        else:
            print("  ❌ Failed to load finance data")
        
        return astro_df is not None and finance_df is not None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run verification."""
    print("\n" + "="*60)
    print("GalaxyScape X Data Verification")
    print("="*60)
    
    files_ok = check_data_files()
    loading_ok = test_data_loading()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Data Files: {'✅ OK' if files_ok else '⚠️  Some missing'}")
    print(f"Data Loading: {'✅ OK' if loading_ok else '❌ Failed'}")
    
    if files_ok and loading_ok:
        print("\n✅ All data checks passed! Ready to test dashboards.")
    else:
        print("\n⚠️  Some issues found. Check output above.")

if __name__ == '__main__':
    main()




