"""
Shared Data Loading Utilities for ML Modules
============================================

LEARNING: Data Engineering Fundamentals
----------------------------------------

This module demonstrates core data engineering concepts:
1. File Discovery: Finding data files across multiple locations
2. Data Normalization: Standardizing column names across datasets
3. Error Handling: Graceful degradation when data is missing
4. Path Resolution: Robust file path construction

DATA ENGINEERING CONCEPTS:
--------------------------

1. FILE DISCOVERY PATTERN:
   ------------------------
   Priority order:
   1. User uploads (highest priority - most recent data)
   2. Dashboard-specific files
   3. General raw data files
   4. Fallback to defaults
   
   Why this order?
   - User uploads = most relevant (user's own data)
   - Dashboard-specific = tailored for specific analysis
   - General files = default datasets
   - Fallbacks = ensure system always works

2. COLUMN NORMALIZATION:
   ---------------------
   Real-world data has inconsistent column names:
   - "name" vs "star_name" vs "Name"
   - "date" vs "Date" vs "timestamp"
   - "ticker" vs "Ticker" vs "Symbol"
   
   Normalization creates consistent interface:
   - Always use same column names internally
   - Handle variations from different data sources
   - Makes downstream code simpler

3. ERROR HANDLING STRATEGY:
   ------------------------
   - Try multiple paths (don't fail on first error)
   - Log errors but continue (defensive programming)
   - Return None gracefully (caller handles missing data)
   - Never crash the application

4. PATH RESOLUTION:
   -----------------
   Use absolute paths relative to project root:
   - Works regardless of current working directory
   - Avoids path-related bugs
   - Makes code portable

LEARNING CHECKPOINT:
-------------------
1. Why check uploads directory first?
   → User's uploaded data is most relevant

2. What is column normalization?
   → Standardizing column names across different data sources

3. Why return None instead of raising exception?
   → Allows graceful degradation (system continues working)
"""
import os
import pandas as pd
from typing import Optional


def load_astronomy_data() -> Optional[pd.DataFrame]:
    """
    Load astronomy CSV data from common raw/upload locations.
    
    LEARNING: This function demonstrates:
    1. Multi-path file discovery (try multiple locations)
    2. Priority-based loading (uploads > raw data)
    3. Column normalization (handle name variations)
    4. Error resilience (try next path if one fails)
    
    Returns:
        DataFrame with astronomy data, or None if no data found
    """
    # STEP 1: DETERMINE BASE DIRECTORY
    # ---------------------------------
    # LEARNING: Get project root directory
    # __file__ = current file path
    # dirname(dirname(...)) = go up two levels to project root
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(backend_dir)
    
    # STEP 2: DEFINE DATA PATHS (Priority Order)
    # -------------------------------------------
    # LEARNING: Priority order matters!
    # 1. Uploads (user's own data - highest priority)
    # 2. Dashboard-specific files
    # 3. General raw data files
    data_paths = [
        os.path.join(project_root, 'data', 'raw', 'astronomy', 'default_astronomy_dataset.csv'),
        os.path.join(project_root, 'data', 'raw', 'astronomy', 'star_explorer.csv'),
        os.path.join(project_root, 'data', 'raw', 'astronomy', 'nasa_exoplanets.csv'),
        os.path.join(backend_dir, 'data', 'raw', 'astronomy', 'star_explorer.csv'),
        os.path.join(project_root, 'uploads', 'astronomy', 'nasa_realistic_stars.csv')
    ]

    # STEP 3: ADD USER UPLOADS (Highest Priority)
    # --------------------------------------------
    # LEARNING: User uploads should override default data
    # Insert at beginning of list (index 0) so they're tried first
    uploads_dir = os.path.join(project_root, 'uploads', 'astronomy')
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            if filename.endswith('.csv'):
                # Insert at beginning (highest priority)
                data_paths.insert(0, os.path.join(uploads_dir, filename))

    # STEP 4: TRY EACH PATH IN ORDER
    # -------------------------------
    # LEARNING: Try multiple paths until one succeeds
    # This provides resilience: if one file is missing, try the next
    for path in data_paths:
        try:
            if os.path.exists(path):
                # STEP 5: LOAD CSV FILE
                # ---------------------
                # LEARNING: pd.read_csv() loads CSV into DataFrame
                # DataFrame = 2D table (rows = observations, columns = features)
                df = pd.read_csv(path)
                
                # STEP 6: COLUMN NORMALIZATION
                # ----------------------------
                # LEARNING: Handle inconsistent column names across datasets
                # Different data sources use different names for same concept
                #
                # Example variations:
                #   - "name" vs "star_name" vs "Name"
                #   - "age" vs "stellar_age" vs "Age"
                #
                # Strategy: Create both names so code can use either
                if 'name' in df.columns and 'star_name' not in df.columns:
                    df['star_name'] = df['name']  # Add normalized name
                if 'star_name' in df.columns and 'name' not in df.columns:
                    df['name'] = df['star_name']  # Add alternative name
                if 'age' in df.columns and 'stellar_age' not in df.columns:
                    df['stellar_age'] = df['age']
                if 'stellar_age' in df.columns and 'age' not in df.columns:
                    df['age'] = df['stellar_age']
                
                # STEP 7: VALIDATE DATA
                # ---------------------
                # LEARNING: Check that data is not empty
                # Empty DataFrame would cause errors downstream
                if len(df) > 0:
                    return df  # Success! Return first valid dataset
        except Exception as exc:  # pragma: no cover - defensive
            # LEARNING: Log error but continue trying other paths
            # Don't crash on one bad file - try the next one
            print(f"Error loading astronomy data from {path}: {exc}")
            continue
    
    # STEP 8: RETURN NONE IF NO DATA FOUND
    # -------------------------------------
    # LEARNING: Return None instead of raising exception
    # Allows caller to handle missing data gracefully
    return None


def load_finance_data() -> Optional[pd.DataFrame]:
    """
    Load finance CSV data from common raw/upload locations.
    
    LEARNING: Similar pattern to astronomy, but with finance-specific:
    1. Date parsing (convert strings to datetime objects)
    2. Ticker normalization (handle Symbol, ticker, Ticker variations)
    3. Price column normalization (close, Close, price variations)
    4. Data validation (require Date, Ticker, Close columns)
    
    Returns:
        DataFrame with finance data, or None if no data found
    """
    # STEP 1: DETERMINE BASE DIRECTORY
    # ---------------------------------
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(backend_dir)
    
    # STEP 2: DEFINE DATA PATHS
    # --------------------------
    data_paths = [
        os.path.join(project_root, 'data', 'raw', 'finance', 'default_finance_dataset.csv'),
        os.path.join(project_root, 'data', 'raw', 'finance', 'risk_dashboard.csv'),
        os.path.join(project_root, 'data', 'raw', 'finance', 'market_data.csv'),
        os.path.join(backend_dir, 'data', 'raw', 'finance', 'market_data.csv'),
        os.path.join(project_root, 'uploads', 'finance', 'market_data.csv'),
        os.path.join(project_root, 'uploads', 'finance', 'market_data_real.csv')
    ]

    # STEP 3: ADD USER UPLOADS (Highest Priority)
    # --------------------------------------------
    uploads_dir = os.path.join(project_root, 'uploads', 'finance')
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            if filename.endswith('.csv'):
                data_paths.insert(0, os.path.join(uploads_dir, filename))

    # STEP 4: TRY EACH PATH
    # ---------------------
    for path in data_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                
                # STEP 5: COLUMN NORMALIZATION (Finance-Specific)
                # ------------------------------------------------
                # LEARNING: Finance data often has inconsistent column names
                # Normalize to standard names: Date, Ticker, Close
                
                # Date column variations
                if 'date' in df.columns and 'Date' not in df.columns:
                    df['Date'] = df['date']
                
                # Ticker/Symbol variations
                if 'ticker' in df.columns and 'Ticker' not in df.columns:
                    df['Ticker'] = df['ticker']
                if 'Symbol' in df.columns and 'Ticker' not in df.columns:
                    df['Ticker'] = df['Symbol']
                
                # Price column variations
                if 'close' in df.columns and 'Close' not in df.columns:
                    df['Close'] = df['close']
                if 'price' in df.columns and 'Close' not in df.columns:
                    df['Close'] = df['price']

                # STEP 6: DATA VALIDATION
                # -----------------------
                # LEARNING: Finance analysis requires specific columns
                # - Date: For time series analysis
                # - Ticker: To identify assets
                # - Close: Price data for calculations
                #
                # If these are missing, data is unusable
                if 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
                    # STEP 7: DATE PARSING
                    # --------------------
                    # LEARNING: Convert date strings to datetime objects
                    # Enables time series operations (sorting, filtering, resampling)
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    # errors='coerce' converts invalid dates to NaT (Not a Time)
                    
                    # STEP 8: REMOVE INVALID ROWS
                    # ---------------------------
                    # LEARNING: Drop rows with missing critical data
                    # Missing Date, Ticker, or Close makes row unusable
                    df = df.dropna(subset=['Date', 'Ticker', 'Close'])
                    
                    # STEP 9: VALIDATE DATA NOT EMPTY
                    # -------------------------------
                    if len(df) > 0:
                        return df  # Success!
        except Exception as exc:  # pragma: no cover - defensive
            # LEARNING: Log error but continue (defensive programming)
            print(f"Error loading finance data from {path}: {exc}")
            continue
    
    return None
