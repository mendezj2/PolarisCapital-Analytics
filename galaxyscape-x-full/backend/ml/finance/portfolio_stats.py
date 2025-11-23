"""
Portfolio Statistics and Modern Portfolio Theory
=================================================

LEARNING: Portfolio Optimization Fundamentals
----------------------------------------------

This module demonstrates Modern Portfolio Theory (MPT), a cornerstone of
financial modeling developed by Harry Markowitz (Nobel Prize 1990).

THEORY: Modern Portfolio Theory (MPT)
-------------------------------------
MPT shows that portfolio risk depends on:
1. Individual asset risks (volatility)
2. Correlations between assets (how they move together)
3. Portfolio weights (how much invested in each asset)

Key Insight: Diversification reduces risk!
- Holding uncorrelated assets reduces portfolio volatility
- Even if individual assets are risky, portfolio can be less risky

MATHEMATICAL FOUNDATION:
------------------------

1. PORTFOLIO EXPECTED RETURN:
   --------------------------
   μₚ = Σᵢ wᵢ × μᵢ
   
   Where:
   - μₚ = Portfolio expected return
   - wᵢ = Weight of asset i (proportion of portfolio)
   - μᵢ = Expected return of asset i
   - Σᵢ = Sum over all assets
   
   LEARNING: Portfolio return is weighted average of individual returns
   Example: 50% in Asset A (10% return) + 50% in Asset B (5% return)
            = 0.5 × 10% + 0.5 × 5% = 7.5%

2. PORTFOLIO VARIANCE:
   -------------------
   σₚ² = Σᵢ Σⱼ wᵢ × wⱼ × σᵢⱼ
   
   Where:
   - σₚ² = Portfolio variance
   - wᵢ, wⱼ = Weights of assets i and j
   - σᵢⱼ = Covariance between assets i and j
   
   In matrix form: σₚ² = wᵀ × Σ × w
   Where:
   - w = Weight vector
   - Σ = Covariance matrix
   - wᵀ = Transpose of w
   
   LEARNING: Portfolio variance depends on:
   - Individual variances (diagonal of covariance matrix)
   - Correlations (off-diagonal elements)
   - Weights (how much in each asset)

3. COVARIANCE vs CORRELATION:
   --------------------------
   Covariance: σᵢⱼ = E[(rᵢ - μᵢ)(rⱼ - μⱼ)]
   Correlation: ρᵢⱼ = σᵢⱼ / (σᵢ × σⱼ)
   
   Relationship: σᵢⱼ = ρᵢⱼ × σᵢ × σⱼ
   
   LEARNING:
   - Covariance: Absolute measure (depends on units)
   - Correlation: Normalized measure (-1 to 1, unitless)
   - Both measure how assets move together

4. SHARPE RATIO:
   -------------
   Sharpe = (μₚ - r_f) / σₚ
   
   Where:
   - μₚ = Portfolio expected return
   - r_f = Risk-free rate (e.g., Treasury bonds)
   - σₚ = Portfolio volatility
   
   LEARNING: Sharpe ratio measures risk-adjusted return
   - Higher Sharpe = better (more return per unit of risk)
   - Used to compare different portfolios

DIVERSIFICATION BENEFIT:
-----------------------
Example: Two assets with same return and volatility
- Asset A: 10% return, 20% volatility
- Asset B: 10% return, 20% volatility
- Correlation: 0.5

Portfolio (50% A, 50% B):
- Return: 10% (same as individual assets)
- Volatility: ~17% (LESS than individual assets!)

LEARNING: Diversification reduces risk without reducing return!

WEIGHT CONSTRAINTS:
------------------
Typically: Σᵢ wᵢ = 1 (weights sum to 100%)
- wᵢ ≥ 0: No short selling (can't have negative weights)
- Or: Allow short selling (wᵢ can be negative)

LEARNING CHECKPOINT:
-------------------
1. Why does portfolio variance depend on correlations?
   → Correlated assets move together, increasing portfolio risk

2. What is the diversification benefit?
   → Combining assets reduces portfolio risk below average individual risk

3. Why use Sharpe ratio instead of just return?
   → Sharpe accounts for risk (return per unit of risk)
"""
import numpy as np
from ml.data_loaders import load_finance_data


def get_portfolio_stats(weights=None, tickers=None):
    """
    Compute portfolio statistics using Modern Portfolio Theory.
    
    LEARNING: This function demonstrates:
    1. Portfolio return calculation (weighted average)
    2. Covariance matrix computation
    3. Portfolio variance calculation (wᵀ × Σ × w)
    4. Sharpe ratio calculation (risk-adjusted return)
    5. Correlation matrix (for visualization)
    
    Args:
        weights: Dict mapping ticker to weight (e.g., {'AAPL': 0.5, 'MSFT': 0.5})
                If None, uses equal weights (1/N for each asset)
        tickers: List of ticker symbols to include
                If None, uses first 10 tickers from data
    
    Returns:
        dict with expected_return, volatility, sharpe_ratio, weights, 
        individual_returns, correlation_matrix
    """
    # STEP 1: DATA LOADING
    # ---------------------
    df = load_finance_data()
    if df is None or len(df) == 0:
        return {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'weights': {},
            'individual_returns': {},
            'correlation_matrix': {}
        }

    # STEP 2: TIME SERIES PREPARATION
    # --------------------------------
    # LEARNING: Sort by ticker and date for proper time series processing
    df = df.sort_values(['Ticker', 'Date'])
    
    # STEP 3: CALCULATE RETURNS
    # --------------------------
    # LEARNING: Returns are the foundation of portfolio analysis
    # Formula: r_t = (P_t - P_{t-1}) / P_{t-1}
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()

    # STEP 4: SELECT TICKERS
    # ----------------------
    # LEARNING: Determine which assets to include in portfolio
    if tickers is None:
        # Use first 10 tickers if not specified
        tickers = df['Ticker'].unique()[:10]
    else:
        # Filter to only tickers that exist in data
        tickers = [t for t in tickers if t in df['Ticker'].unique()]

    if len(tickers) == 0:
        return {
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'weights': {},
            'individual_returns': {},
            'correlation_matrix': {}
        }

    # STEP 5: PIVOT TO WIDE FORMAT
    # -----------------------------
    # LEARNING: Transform to matrix format for portfolio calculations
    # Pivot creates: Date as rows, Ticker as columns, Close as values
    df_pivot = df[df['Ticker'].isin(tickers)].pivot(index='Date', columns='Ticker', values='Close')
    
    # STEP 6: CALCULATE RETURNS MATRIX
    # ---------------------------------
    # LEARNING: Returns matrix has same structure as price matrix
    # Each column = time series of returns for one asset
    returns_matrix = df_pivot.pct_change().dropna()
    # Shape: (n_dates, n_tickers)

    # STEP 7: CALCULATE INDIVIDUAL ASSET RETURNS
    # ------------------------------------------
    # LEARNING: Expected return for each asset (annualized)
    # Formula: μᵢ = mean(daily_returns) × 252
    individual_returns = {}
    for ticker in tickers:
        ticker_returns = returns_matrix[ticker].dropna()
        # Annualized mean return
        individual_returns[ticker] = float(ticker_returns.mean() * 252) if len(ticker_returns) > 0 else 0.0

    # STEP 8: COMPUTE CORRELATION MATRIX
    # ----------------------------------
    # LEARNING: Correlation measures how assets move together
    # Used for visualization and understanding relationships
    corr_matrix = returns_matrix.corr()
    # Shape: (n_tickers, n_tickers), symmetric matrix
    
    # Convert to nested dict for JSON serialization
    correlation_dict = {
        t1: {t2: float(corr_matrix.loc[t1, t2]) for t2 in tickers if t1 in corr_matrix.index and t2 in corr_matrix.columns}
        for t1 in tickers
    }

    # STEP 9: COMPUTE COVARIANCE MATRIX
    # ----------------------------------
    # LEARNING: Covariance matrix is needed for portfolio variance calculation
    # Covariance: σᵢⱼ = E[(rᵢ - μᵢ)(rⱼ - μⱼ)]
    #
    # Annualized: Multiply daily covariance by 252
    # Why? Variance scales linearly with time
    cov_matrix = returns_matrix.cov() * 252
    # Shape: (n_tickers, n_tickers), symmetric matrix

    # STEP 10: DETERMINE PORTFOLIO WEIGHTS
    # ------------------------------------
    # LEARNING: Weights determine how much of portfolio is in each asset
    # Constraint: Σᵢ wᵢ = 1 (weights sum to 100%)
    if weights is None:
        # Equal weights: 1/N for each asset
        # LEARNING: Equal weights = naive diversification
        weight_val = 1.0 / len(tickers)
        weights = {ticker: weight_val for ticker in tickers}
    else:
        # Normalize weights to sum to 1
        # LEARNING: If user provides weights that don't sum to 1, normalize them
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()} if total > 0 else weights

    # STEP 11: CALCULATE PORTFOLIO EXPECTED RETURN
    # ---------------------------------------------
    # LEARNING: Portfolio return = weighted average of individual returns
    # Formula: μₚ = Σᵢ wᵢ × μᵢ
    #
    # Example: 50% in Asset A (10% return) + 50% in Asset B (5% return)
    #          = 0.5 × 10% + 0.5 × 5% = 7.5%
    mu_p = sum(weights[ticker] * individual_returns[ticker] for ticker in tickers)

    # STEP 12: CALCULATE PORTFOLIO VARIANCE
    # -------------------------------------
    # LEARNING: Portfolio variance depends on:
    # 1. Individual asset variances (diagonal of covariance matrix)
    # 2. Covariances between assets (off-diagonal elements)
    # 3. Portfolio weights
    
    # Convert weights to vector
    w_vec = np.array([weights.get(ticker, 0.0) for ticker in tickers])
    
    # LEARNING: Portfolio variance formula: σₚ² = wᵀ × Σ × w
    # In expanded form: σₚ² = Σᵢ Σⱼ wᵢ × wⱼ × σᵢⱼ
    #
    # This double sum considers:
    # - i = j: Variance terms (wᵢ² × σᵢ²)
    # - i ≠ j: Covariance terms (wᵢ × wⱼ × σᵢⱼ)
    sigma_p_squared = 0.0
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if t1 in cov_matrix.index and t2 in cov_matrix.columns:
                # LEARNING: Each term contributes wᵢ × wⱼ × σᵢⱼ
                sigma_p_squared += w_vec[i] * w_vec[j] * cov_matrix.loc[t1, t2]
    
    # STEP 13: CALCULATE PORTFOLIO VOLATILITY
    # ---------------------------------------
    # LEARNING: Volatility = standard deviation = √variance
    # Take square root to get volatility (not variance)
    sigma_p = np.sqrt(max(0, sigma_p_squared))  # max(0, ...) ensures non-negative

    # STEP 14: CALCULATE SHARPE RATIO
    # --------------------------------
    # LEARNING: Sharpe ratio = risk-adjusted return
    # Formula: Sharpe = (μₚ - r_f) / σₚ
    #
    # In this simplified version, we assume r_f = 0 (risk-free rate)
    # Full formula would subtract risk-free rate
    #
    # Interpretation:
    # - Higher Sharpe = better (more return per unit of risk)
    # - Used to compare different portfolios
    sharpe = mu_p / sigma_p if sigma_p > 0 else 0.0

    # LEARNING: Results interpretation:
    # - expected_return: Portfolio's expected annual return
    # - volatility: Portfolio's annual volatility (risk)
    # - sharpe_ratio: Risk-adjusted return (higher is better)
    # - weights: How much invested in each asset
    # - individual_returns: Expected return for each asset
    # - correlation_matrix: How assets move together
    
    return {
        'expected_return': float(mu_p),
        'volatility': float(sigma_p),
        'sharpe_ratio': float(sharpe),
        'weights': {str(k): float(v) for k, v in weights.items()},
        'individual_returns': {str(k): float(v) for k, v in individual_returns.items()},
        'correlation_matrix': {str(k): {str(k2): float(v2) for k2, v2 in v.items()} for k, v in correlation_dict.items()}
    }
