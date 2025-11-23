"""
Rolling/Streaming Metrics for Real-Time Risk Monitoring
========================================================

LEARNING: Time Series Analysis and Rolling Windows
---------------------------------------------------

This module demonstrates rolling window calculations, a fundamental technique
for time series analysis and real-time monitoring.

THEORY: What are Rolling Windows?
---------------------------------
A rolling window is a fixed-size window that moves through time series data:

Example: 30-day rolling window
- Day 1-30: Calculate metric (e.g., volatility)
- Day 2-31: Move window forward, recalculate
- Day 3-32: Move again, recalculate
- And so on...

Why use rolling windows?
- Captures time-varying patterns (volatility changes over time)
- Smooths out noise (averages over multiple periods)
- Enables real-time monitoring (update as new data arrives)

ROLLING VOLATILITY:
------------------
Volatility changes over time. Rolling volatility tracks this:

    σ_t(rolling) = std(returns[t-w:t]) × √252

Where:
    - w = window size (e.g., 30 days)
    - returns[t-w:t] = returns in window ending at time t
    - √252 = annualization factor

Interpretation:
- High rolling volatility = recent period was volatile
- Low rolling volatility = recent period was calm
- Increasing trend = volatility is rising (risk increasing)

ROLLING CORRELATION:
-------------------
Correlation between assets also changes over time:

    ρ_t(A,B) = corr(returns_A[t-w:t], returns_B[t-w:t])

Why it changes:
- Market regimes (bull vs bear markets)
- Sector rotations (different sectors perform at different times)
- Crisis periods (correlations spike during crashes)

LEARNING: During market crashes, correlations increase!
- Normal times: Assets somewhat independent
- Crisis: Everything moves together (correlation → 1.0)
- This is called "correlation breakdown" or "contagion"

DATA ENGINEERING CONCEPTS:
--------------------------
1. Time-based Filtering: Select recent data (last N days)
2. Rolling Calculations: Apply function to sliding window
3. Time Series Alignment: Merge data from different assets on same dates
4. Data Aggregation: Combine metrics across multiple assets

LEARNING CHECKPOINT:
-------------------
1. Why use rolling windows instead of all historical data?
   → Recent data is more relevant; old data may be outdated

2. What does increasing rolling volatility indicate?
   → Risk is rising (market becoming more volatile)

3. Why do correlations increase during crises?
   → Assets move together when markets panic (contagion effect)
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from ml.data_loaders import load_finance_data


def get_streaming_metrics(df=None, window_days=30):
    """
    Compute rolling volatility, correlation, and latest risk scores.
    
    LEARNING: This function demonstrates:
    1. Time-based data filtering (recent window)
    2. Rolling window calculations (sliding window statistics)
    3. Time series alignment (matching dates across assets)
    4. Real-time risk monitoring (latest risk scores)
    
    Args:
        window_days: Size of rolling window in days (e.g., 30 = last 30 days)
                    - Smaller window = more responsive to recent changes
                    - Larger window = smoother, less noisy
    
    Returns:
        dict with timestamps, rolling_volatility, rolling_correlation, latest_risk_scores
    """
    # STEP 1: DATA LOADING
    # ---------------------
    if df is None:
        df = load_finance_data()
    if df is None or len(df) == 0:
        return {
            'timestamps': [],
            'rolling_volatility': [],
            'rolling_correlation': {},
            'latest_risk_scores': {}
        }

    # STEP 2: TIME SERIES PREPARATION
    # --------------------------------
    # LEARNING: Sort by ticker and date for proper time series processing
    df = df.sort_values(['Ticker', 'Date'])
    
    # STEP 3: CALCULATE RETURNS
    # --------------------------
    # LEARNING: Returns are the foundation of risk metrics
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()

    # STEP 4: TIME-BASED FILTERING
    # -----------------------------
    # LEARNING: Select recent data for rolling calculations
    # Only use data from last window_days for efficiency and relevance
    latest_date = df['Date'].max()  # Most recent date in dataset
    window_start = latest_date - timedelta(days=window_days)  # Go back N days
    recent_df = df[df['Date'] >= window_start]  # Filter to recent window
    
    # LEARNING: Why filter to recent data?
    # - Efficiency: Less data to process
    # - Relevance: Recent data more predictive of current risk
    # - Real-time: Mimics streaming data scenario

    # STEP 5: SELECT ASSETS FOR ANALYSIS
    # ----------------------------------
    # LEARNING: Limit to top 10 tickers for performance
    # In production, might analyze all assets or user-selected subset
    tickers = recent_df['Ticker'].unique()[:10]
    
    # STEP 6: CALCULATE ROLLING VOLATILITY
    # ------------------------------------
    # LEARNING: Rolling volatility tracks how risk changes over time
    rolling_vol = []
    timestamps = []

    for ticker in tickers:
        # Filter to current ticker and sort by date
        ticker_df = recent_df[recent_df['Ticker'] == ticker].sort_values('Date')
        
        if len(ticker_df) < 5:  # Need minimum data points
            continue

        returns = ticker_df['returns'].dropna()
        if len(returns) >= 5:
            # STEP 7: ROLLING STANDARD DEVIATION
            # -----------------------------------
            # LEARNING: rolling() creates sliding window
            # window=10 means: calculate std of last 10 returns
            # As window moves forward, get new std for each time point
            #
            # Example with window=5:
            #   Day 1-5: std(returns[1:5])
            #   Day 2-6: std(returns[2:6])
            #   Day 3-7: std(returns[3:7])
            #   ...
            rolling_std = returns.rolling(window=min(10, len(returns))).std()
            
            # LEARNING: Annualize volatility
            # Daily std → Annual std: σ_annual = σ_daily × √252
            rolling_std_annual = rolling_std * np.sqrt(252)
            valid = rolling_std_annual.dropna()
            
            # Store results (drop NaN from initial window)
            rolling_vol.extend(valid.tolist())
            timestamps.extend(valid.index.tolist())

    # STEP 8: DEDUPLICATE AND SORT TIMESTAMPS
    # ----------------------------------------
    # LEARNING: Multiple tickers may have same dates
    # Create DataFrame to deduplicate and sort
    if timestamps:
        df_vol = pd.DataFrame({'timestamp': timestamps, 'volatility': rolling_vol})
        df_vol = df_vol.drop_duplicates('timestamp').sort_values('timestamp')
        df_vol['timestamp'] = pd.to_datetime(df_vol['timestamp'], errors='coerce')
        timestamps = df_vol['timestamp'].dt.strftime('%Y-%m-%d').tolist()
        rolling_vol = df_vol['volatility'].tolist()

    # STEP 9: CALCULATE ROLLING CORRELATION
    # --------------------------------------
    # LEARNING: Correlation between two assets over rolling window
    # Shows how relationship changes over time
    rolling_corr = {}
    
    if len(tickers) >= 2:
        # Use first two tickers for correlation calculation
        ticker1, ticker2 = tickers[0], tickers[1]
        ticker1_df = recent_df[recent_df['Ticker'] == ticker1].sort_values('Date')
        ticker2_df = recent_df[recent_df['Ticker'] == ticker2].sort_values('Date')

        # STEP 10: TIME SERIES ALIGNMENT
        # --------------------------------
        # LEARNING: Merge on Date to align time series
        # Both assets must have data on same dates for correlation
        merged = pd.merge(
            ticker1_df[['Date', 'returns']],
            ticker2_df[['Date', 'returns']],
            on='Date',
            suffixes=('_1', '_2')
        ).dropna()  # Remove rows where either asset missing data
        
        # LEARNING: Result has columns: Date, returns_1, returns_2
        # Now we can calculate correlation between aligned returns

        if len(merged) >= 10:
            # STEP 11: ROLLING CORRELATION
            # -----------------------------
            # LEARNING: Calculate correlation over rolling window
            # window=10 means: correlation of last 10 aligned returns
            window = min(10, len(merged))
            
            # LEARNING: rolling().corr() calculates correlation in each window
            # Returns series of correlation values (one per time point)
            rolling_corr_val = merged['returns_1'].rolling(window=window).corr(merged['returns_2']).iloc[-1]
            # iloc[-1] gets most recent correlation value
            
            rolling_corr[f"{ticker1}_{ticker2}"] = float(rolling_corr_val) if not np.isnan(rolling_corr_val) else 0.0

    # STEP 12: CALCULATE LATEST RISK SCORES
    # --------------------------------------
    # LEARNING: Current risk level for each asset
    # Uses all recent data (not rolling window) for latest snapshot
    latest_risk_scores = {}
    for ticker in tickers:
        ticker_df = recent_df[recent_df['Ticker'] == ticker]
        if len(ticker_df) > 0:
            returns = ticker_df['returns'].dropna()
            if len(returns) > 0:
                # Calculate volatility from all recent returns
                volatility = returns.std() * np.sqrt(252)  # Annualized
                
                # LEARNING: Convert to risk score (0-100 scale)
                # Cap at 100 to prevent extreme values
                risk_score = min(100, volatility * 100)
                latest_risk_scores[str(ticker)] = float(risk_score)
    
    # LEARNING: Results interpretation:
    # - timestamps: Time points for rolling volatility chart
    # - rolling_volatility: Volatility values (shows how risk changes over time)
    # - rolling_correlation: How two assets move together (changes over time)
    # - latest_risk_scores: Current risk level for each asset (snapshot)
    
    return {
        'timestamps': timestamps[:100],  # Limit to 100 points for performance
        'rolling_volatility': rolling_vol[:100],
        'rolling_correlation': rolling_corr,
        'latest_risk_scores': latest_risk_scores
    }
