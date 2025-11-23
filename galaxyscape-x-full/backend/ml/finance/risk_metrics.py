"""
Risk Metrics for Financial Analysis
====================================

LEARNING: Financial Risk Modeling Fundamentals
----------------------------------------------

This module demonstrates core financial risk metrics used in portfolio management,
risk assessment, and regulatory compliance.

THEORY: Why Measure Risk?
-------------------------
In finance, risk = uncertainty of returns. Investors need to quantify:
1. Volatility: How much prices fluctuate (standard deviation of returns)
2. Sharpe Ratio: Risk-adjusted return (reward per unit of risk)
3. Maximum Drawdown: Worst peak-to-trough decline (pain during crashes)
4. Risk Score: Composite metric combining multiple risk factors

MATHEMATICAL FOUNDATIONS:
------------------------

1. VOLATILITY (Annualized Standard Deviation):
   -------------------------------------------
   σ_annual = σ_daily × √252
   
   Where:
   - σ_daily = standard deviation of daily returns
   - 252 = trading days per year (approximately)
   - √252 ≈ 15.87 (annualization factor)
   
   Why annualize? To compare assets with different time horizons.
   
   Interpretation:
   - σ = 0.20 (20%): Moderate volatility (typical for stocks)
   - σ = 0.40 (40%): High volatility (risky assets)
   - σ = 0.10 (10%): Low volatility (bonds, stable stocks)

2. SHARPE RATIO (Risk-Adjusted Return):
   -------------------------------------
   Sharpe = (R_p - R_f) / σ_p
   
   Where:
   - R_p = Portfolio return (annualized)
   - R_f = Risk-free rate (e.g., Treasury bonds, typically 2-3%)
   - σ_p = Portfolio volatility (annualized)
   
   Interpretation:
   - Sharpe > 1.0: Good risk-adjusted return
   - Sharpe > 2.0: Excellent (rare)
   - Sharpe < 0.5: Poor (risk not compensated by return)
   
   LEARNING: Sharpe ratio answers: "Is the extra return worth the extra risk?"

3. MAXIMUM DRAWDOWN (Worst Case Loss):
   ------------------------------------
   Drawdown at time t = (Peak - Current) / Peak
   
   Maximum Drawdown = max(drawdown over all time periods)
   
   Why it matters:
   - Shows worst-case scenario (how much you could lose)
   - Critical for risk management and position sizing
   - Used in regulatory capital requirements
   
   Example:
   - Stock peaks at $100, drops to $70
   - Drawdown = (100 - 70) / 100 = 30%
   - If it later drops to $60, max drawdown = 40%

4. RISK SCORE (Composite Metric):
   ------------------------------
   Risk Score = Volatility Score + Drawdown Score
   
   Where each component is capped at 50 points (total max = 100)
   
   This creates a 0-100 scale:
   - 0-30: Low risk (conservative investments)
   - 30-60: Moderate risk (balanced portfolio)
   - 60-100: High risk (aggressive investments)

DATA ENGINEERING CONCEPTS:
--------------------------
1. GroupBy Operations: Calculate metrics per ticker (asset)
2. Time Series Processing: Sort by date, calculate returns
3. Rolling Calculations: Expanding windows for cumulative metrics
4. Data Validation: Check minimum data requirements

LEARNING CHECKPOINT:
-------------------
1. Why multiply daily volatility by √252?
   → To annualize (convert daily to yearly scale)

2. What does Sharpe ratio = 1.5 mean?
   → For every 1% of risk, you get 1.5% of excess return

3. Why is max drawdown important?
   → It shows the worst-case loss, critical for risk management
"""
import numpy as np
from ml.data_loaders import load_finance_data


def get_risk_metrics(df=None):
    """
    Compute comprehensive risk metrics for financial assets.
    
    LEARNING: This function demonstrates:
    1. Time series data processing (sorting by date)
    2. Return calculation (percentage change)
    3. Statistical aggregation (groupby operations)
    4. Financial metric computation (volatility, Sharpe, drawdown)
    5. Composite scoring (risk score calculation)
    
    Returns:
        dict with tickers, volatility, sharpe_ratio, max_drawdown, risk_scores
    """
    # STEP 1: DATA LOADING
    # ---------------------
    if df is None:
        df = load_finance_data()

    if df is None or len(df) == 0:
        return {
            'tickers': [],
            'volatility': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'risk_scores': []
        }

    # STEP 2: DATA PREPROCESSING - Time Series Sorting
    # --------------------------------------------------
    # LEARNING: Financial data must be sorted chronologically for time series analysis
    # Groupby('Ticker') ensures we process each asset separately
    df = df.sort_values(['Ticker', 'Date'])
    
    # STEP 3: FEATURE ENGINEERING - Calculate Returns
    # ------------------------------------------------
    # LEARNING: Returns = percentage change in price
    # Formula: r_t = (P_t - P_{t-1}) / P_{t-1}
    # 
    # Why returns instead of prices?
    # - Returns are stationary (mean and variance don't change over time)
    # - Prices are non-stationary (trend upward over time)
    # - Returns are comparable across different price levels
    df['returns'] = df.groupby('Ticker')['Close'].pct_change()
    # pct_change() calculates: (current - previous) / previous

    # STEP 4: ITERATE OVER TICKERS (Assets)
    # --------------------------------------
    # LEARNING: Process each asset independently
    # Limit to 20 tickers for performance (can be adjusted)
    tickers = df['Ticker'].unique()[:20]
    result = {'tickers': [], 'volatility': [], 'sharpe_ratio': [], 'max_drawdown': [], 'risk_scores': []}

    for ticker in tickers:
        # STEP 5: FILTER DATA FOR CURRENT TICKER
        # ----------------------------------------
        ticker_df = df[df['Ticker'] == ticker].copy()
        
        # LEARNING: Minimum data requirement for reliable statistics
        # Need at least 10 data points for meaningful calculations
        if len(ticker_df) < 10:
            continue

        # STEP 6: EXTRACT RETURNS AND VALIDATE
        # -------------------------------------
        returns = ticker_df['returns'].dropna()  # Remove NaN (first row has no previous value)
        
        if len(returns) < 5:
            continue  # Need minimum returns for calculation

        # STEP 7: CALCULATE VOLATILITY (Annualized)
        # -------------------------------------------
        # LEARNING: Volatility = standard deviation of returns
        # Daily volatility: σ_daily = std(returns)
        # Annualized volatility: σ_annual = σ_daily × √252
        #
        # Why √252? 
        # - Variance scales linearly with time: Var(annual) = 252 × Var(daily)
        # - Standard deviation scales with √time: σ_annual = √252 × σ_daily
        # - 252 ≈ trading days per year (365 - weekends - holidays)
        volatility = returns.std() * np.sqrt(252)
        
        # STEP 8: CALCULATE MEAN RETURN (Annualized)
        # -------------------------------------------
        # LEARNING: Mean return also needs annualization
        # Daily mean: μ_daily = mean(returns)
        # Annualized mean: μ_annual = μ_daily × 252
        mean_return = returns.mean() * 252

        # STEP 9: CALCULATE SHARPE RATIO
        # --------------------------------
        # LEARNING: Sharpe Ratio = (Return - RiskFreeRate) / Volatility
        # 
        # In this simplified version, we assume risk-free rate = 0
        # Full formula: Sharpe = (μ - r_f) / σ
        #
        # Interpretation:
        # - Higher Sharpe = better risk-adjusted return
        # - Negative Sharpe = returns don't compensate for risk
        sharpe = mean_return / volatility if volatility > 0 else 0.0

        # STEP 10: CALCULATE MAXIMUM DRAWDOWN
        # ------------------------------------
        # LEARNING: Drawdown measures peak-to-trough decline
        #
        # Algorithm:
        # 1. Calculate cumulative returns: (1 + r₁)(1 + r₂)...(1 + rₙ)
        # 2. Track running maximum (peak)
        # 3. Calculate drawdown = (peak - current) / peak
        # 4. Find maximum drawdown
        
        # Cumulative product: converts returns to cumulative wealth
        # Example: [0.01, 0.02, -0.01] → [1.01, 1.0302, 1.019898]
        cumulative = (1 + returns).cumprod()
        
        # LEARNING: Expanding window finds running maximum
        # At each point, track the highest value seen so far
        running_max = cumulative.expanding().max()
        
        # LEARNING: Drawdown = percentage decline from peak
        # Formula: (current - peak) / peak
        # Negative values mean we're below the peak
        drawdown = (cumulative - running_max) / running_max
        
        # Maximum drawdown = worst decline (most negative, so we take absolute value)
        max_dd = abs(drawdown.min())

        # STEP 11: CALCULATE COMPOSITE RISK SCORE
        # ----------------------------------------
        # LEARNING: Combine multiple risk factors into single score
        #
        # Strategy: Cap each component at 50 points
        # - Volatility score: min(50, volatility × 100)
        # - Drawdown score: min(50, max_dd × 100)
        # - Total risk: 0-100 scale
        #
        # Why cap at 50? Prevents one metric from dominating
        vol_score = min(50, volatility * 100)  # Convert to percentage, cap at 50
        dd_score = min(50, max_dd * 100)  # Convert to percentage, cap at 50
        risk_score = vol_score + dd_score  # Combined score (0-100)

        # STEP 12: STORE RESULTS
        # -----------------------
        result['tickers'].append(str(ticker))
        result['volatility'].append(float(volatility))
        result['sharpe_ratio'].append(float(sharpe))
        result['max_drawdown'].append(float(max_dd))
        result['risk_scores'].append(float(risk_score))

    return result
