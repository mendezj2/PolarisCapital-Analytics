"""
Correlation Network for Financial Market Analysis
==================================================

LEARNING: Network Science in Finance
------------------------------------

This module demonstrates how to build correlation networks from financial data,
revealing relationships and dependencies between assets.

THEORY: What is a Correlation Network?
--------------------------------------
A correlation network is a graph where:
- Nodes = Financial assets (stocks, bonds, etc.)
- Edges = Correlation between assets (how they move together)
- Edge weight = Strength of correlation (|correlation coefficient|)

Why build correlation networks?
- Identify market sectors (highly correlated assets)
- Detect systemic risk (contagion effects)
- Portfolio diversification (low correlation = better diversification)
- Market structure analysis (clusters, communities)

CORRELATION COEFFICIENT:
-----------------------
Pearson correlation measures linear relationship:

    ρ(X,Y) = Cov(X,Y) / (σₓ × σᵧ)

Where:
    - Cov(X,Y) = Covariance (how X and Y vary together)
    - σₓ, σᵧ = Standard deviations of X and Y

Interpretation:
    - ρ = 1.0: Perfect positive correlation (move together)
    - ρ = 0.0: No correlation (independent)
    - ρ = -1.0: Perfect negative correlation (move opposite)
    - |ρ| > 0.7: Strong correlation
    - |ρ| < 0.3: Weak correlation

RETURNS vs PRICES:
-----------------
We calculate correlation on returns, not prices:

Why returns?
- Returns are stationary (mean and variance don't change)
- Prices are non-stationary (trend upward over time)
- Returns are comparable across different price levels

Formula: r_t = (P_t - P_{t-1}) / P_{t-1}

NETWORK THRESHOLD:
-----------------
We only create edges for |correlation| ≥ threshold

Why use a threshold?
- Reduces visual clutter (too many edges = unreadable)
- Focuses on meaningful relationships
- Typical threshold: 0.5-0.7 (moderate to strong correlation)

NETWORK METRICS:
---------------
Node properties:
- Volatility: Risk measure (standard deviation of returns)
- Sector: Industry classification
- Degree: Number of connections (high = central to network)

Edge properties:
- Correlation: Strength of relationship (-1 to 1)
- Weight: Absolute correlation (0 to 1, for visualization)

DATA ENGINEERING CONCEPTS:
-------------------------
1. Pivot Tables: Transform long format to wide format
   - Long: Date, Ticker, Close
   - Wide: Date as index, Ticker as columns, Close as values
   
2. Time Series Alignment: Ensure all tickers have same dates
   - Missing dates create NaN values
   - Drop rows with insufficient data

3. Correlation Matrix: Pairwise correlations between all assets
   - Symmetric matrix (corr(A,B) = corr(B,A))
   - Diagonal = 1.0 (asset perfectly correlated with itself)

LEARNING CHECKPOINT:
-------------------
1. Why calculate correlation on returns, not prices?
   → Returns are stationary; prices trend upward over time

2. What does correlation = 0.8 mean?
   → Strong positive relationship (assets move together 80% of the time)

3. Why use a threshold for edges?
   → Reduces clutter and focuses on meaningful relationships
"""
import numpy as np
from ml.data_loaders import load_finance_data


def get_correlation_network(df=None, threshold=0.5):
    """
    Build correlation network graph from financial data.
    
    LEARNING: This function demonstrates:
    1. Time series data preparation (sorting, pivoting)
    2. Return calculation (percentage change)
    3. Correlation matrix computation
    4. Network graph construction (nodes and edges)
    5. Threshold filtering (only strong correlations)
    
    Args:
        df: Historical price data (Date, Ticker, Close columns)
        threshold: Minimum |correlation| to create edge (0.0 to 1.0)
                  - 0.5 = moderate correlation
                  - 0.7 = strong correlation
                  - Lower = more edges (more connections)
                  - Higher = fewer edges (only strong relationships)
    
    Returns:
        dict with nodes (assets) and edges (correlations)
    """
    # STEP 1: DATA LOADING
    # ---------------------
    if df is None:
        df = load_finance_data()

    if df is None or len(df) == 0:
        return {'nodes': [], 'edges': []}

    # STEP 2: TIME SERIES PREPARATION
    # --------------------------------
    # LEARNING: Sort by ticker and date for proper time series processing
    # This ensures chronological order for each asset
    df = df.sort_values(['Ticker', 'Date'])
    
    # STEP 3: PIVOT TABLE TRANSFORMATION
    # -----------------------------------
    # LEARNING: Transform from "long" to "wide" format
    #
    # Before (long format):
    #   Date       Ticker  Close
    #   2024-01-01 AAPL    150.0
    #   2024-01-01 MSFT    300.0
    #   2024-01-02 AAPL    151.0
    #   2024-01-02 MSFT    301.0
    #
    # After (wide format):
    #   Date       AAPL    MSFT
    #   2024-01-01 150.0   300.0
    #   2024-01-02 151.0   301.0
    #
    # Why pivot? Correlation needs aligned time series (same dates for all assets)
    df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
    # Result: Date as rows, Ticker as columns, Close as values

    # STEP 4: CALCULATE RETURNS
    # --------------------------
    # LEARNING: Correlation is calculated on returns, not prices
    #
    # Why returns?
    # - Returns are stationary (mean and variance constant)
    # - Prices are non-stationary (trend upward)
    # - Returns are comparable across different price levels
    #
    # Formula: r_t = (P_t - P_{t-1}) / P_{t-1}
    returns = df_pivot.pct_change().dropna()
    # dropna() removes first row (no previous value to calculate return)

    # STEP 5: COMPUTE CORRELATION MATRIX
    # -----------------------------------
    # LEARNING: Correlation matrix shows pairwise relationships
    #
    # Result is symmetric matrix:
    #        AAPL  MSFT  GOOGL
    # AAPL   1.0   0.7   0.6
    # MSFT   0.7   1.0   0.8
    # GOOGL  0.6   0.8   1.0
    #
    # Diagonal = 1.0 (asset perfectly correlated with itself)
    # Off-diagonal = correlation between different assets
    corr_matrix = returns.corr()
    # Shape: (n_tickers, n_tickers)

    # STEP 6: CREATE NODES (ASSETS)
    # -----------------------------
    # LEARNING: Limit to top 20 tickers for performance and visualization clarity
    tickers = corr_matrix.columns.tolist()[:20]
    nodes = []
    
    for ticker in tickers:
        # STEP 7: CALCULATE NODE PROPERTIES
        # ----------------------------------
        # LEARNING: Each node has properties for visualization
        ticker_data = df[df['Ticker'] == ticker]
        
        if len(ticker_data) > 0:
            # Calculate volatility (risk measure)
            # LEARNING: Volatility = standard deviation of returns, annualized
            returns_ticker = ticker_data['Close'].pct_change().dropna()
            
            # Annualized volatility: σ_annual = σ_daily × √252
            volatility = returns_ticker.std() * np.sqrt(252) if len(returns_ticker) > 0 else 0.0
            
            # LEARNING: Node properties:
            # - id: Unique identifier (ticker symbol)
            # - name: Display name
            # - volatility: Risk measure (for node size in visualization)
            # - sector: Industry classification (for node color in visualization)
            nodes.append({
                'id': str(ticker),
                'name': str(ticker),
                'volatility': float(volatility),
                'sector': ticker_data['Sector'].iloc[0] if 'Sector' in ticker_data.columns else 'Unknown'
            })

    # STEP 8: CREATE EDGES (CORRELATIONS)
    # ------------------------------------
    # LEARNING: Edges represent relationships between assets
    # Only create edges for correlations above threshold
    
    edges = []
    
    # LEARNING: Iterate through all pairs (avoid duplicates)
    # For each pair (ticker1, ticker2), calculate correlation
    for i, ticker1 in enumerate(tickers):
        for ticker2 in tickers[i+1:]:  # i+1 avoids duplicate pairs (A-B and B-A)
            # Get correlation from matrix
            corr = corr_matrix.loc[ticker1, ticker2]
            
            # STEP 9: THRESHOLD FILTERING
            # ----------------------------
            # LEARNING: Only create edge if |correlation| ≥ threshold
            # abs() because both positive and negative correlations are meaningful
            # - Positive: Assets move together (both up or both down)
            # - Negative: Assets move opposite (one up, one down)
            if abs(corr) >= threshold:
                edges.append({
                    'source': str(ticker1),      # Source node
                    'target': str(ticker2),      # Target node
                    'correlation': float(corr),   # Correlation value (-1 to 1)
                    'weight': float(abs(corr))    # Edge weight for visualization (0 to 1)
                })
    
    # LEARNING: Network interpretation:
    # - Dense network (many edges): Highly correlated market (systemic risk)
    # - Sparse network (few edges): Diversified market (low correlation)
    # - Clusters: Groups of highly correlated assets (sectors, industries)
    
    return {'nodes': nodes, 'edges': edges}
