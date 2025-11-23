"""
Monte Carlo Simulation for Future Portfolio Outcomes
=====================================================

LEARNING: Monte Carlo Methods in Financial Modeling
----------------------------------------------------

This module demonstrates Monte Carlo simulation, a powerful technique for modeling
uncertainty and generating probabilistic forecasts.

THEORY: What is Monte Carlo Simulation?
---------------------------------------
Monte Carlo simulation uses random sampling to model uncertain outcomes. Instead of
predicting a single future value, it generates thousands of possible scenarios.

Key Concepts:
1. Random Sampling: Generate random numbers from probability distributions
2. Scenario Generation: Each random sample = one possible future
3. Statistical Aggregation: Analyze distribution of outcomes (percentiles, mean)

WHY MONTE CARLO?
---------------
Financial markets are uncertain. We can't predict exact future prices, but we can:
- Model the probability distribution of returns
- Generate many possible scenarios
- Calculate confidence intervals (e.g., "95% chance value will be between X and Y")

MATHEMATICAL FOUNDATION:
------------------------

1. GEOMETRIC BROWNIAN MOTION (GBM):
   ---------------------------------
   This is the standard model for stock price evolution:
   
   S(t) = S₀ × exp((μ - ½σ²)t + σ√t × Z)
   
   Where:
   - S(t) = Price at time t
   - S₀ = Initial price
   - μ = Expected return (drift)
   - σ = Volatility (standard deviation)
   - t = Time horizon
   - Z = Random variable from standard normal distribution (N(0,1))
   
   LEARNING: Why (μ - ½σ²)?
   - This is the "drift adjustment" for log-normal distribution
   - Without it, expected value would be S₀ × exp(μt)
   - With it, expected value = S₀ × exp(μt) (correct!)
   
   The term -½σ² comes from Ito's Lemma in stochastic calculus.

2. LOG-NORMAL DISTRIBUTION:
   -------------------------
   Stock prices follow log-normal distribution because:
   - Prices can't be negative (log ensures positive values)
   - Returns are normally distributed
   - Price = exp(return), so price is log-normal
   
   Properties:
   - Mean: S₀ × exp(μt)
   - Variance: S₀² × exp(2μt) × (exp(σ²t) - 1)

3. PERCENTILES (Confidence Intervals):
   ------------------------------------
   After generating N scenarios, we sort them and find percentiles:
   
   - p5 (5th percentile): 5% of scenarios below this value (pessimistic)
   - p25 (25th percentile): 25% below (lower quartile)
   - p50 (50th percentile): Median (50% below, 50% above)
   - p75 (75th percentile): 75% below (upper quartile)
   - p95 (95th percentile): 95% below (optimistic)
   
   LEARNING: These create "fan charts" showing uncertainty bands

DATA ENGINEERING CONCEPTS:
--------------------------
1. Time Series Aggregation: Calculate μ and σ from historical returns
2. Statistical Estimation: Use sample statistics to estimate population parameters
3. Random Number Generation: np.random.randn() generates N(0,1) samples
4. Vectorization: Process many scenarios efficiently with NumPy

LEARNING CHECKPOINT:
-------------------
1. Why use exp() in the GBM formula?
   → To ensure prices stay positive (log-normal distribution)

2. What does p95 = $15,000 mean?
   → 95% of scenarios result in value ≤ $15,000 (optimistic bound)

3. Why do we need many scenarios (n_scenarios=1000)?
   → More scenarios = better approximation of true distribution
"""
import numpy as np
from scipy.stats import norm
from ml.data_loaders import load_finance_data


def get_future_outcomes(df=None, initial_value=100000, time_horizon_years=1, n_scenarios=1000):
    """
    Project future portfolio values using Monte Carlo simulation.
    
    LEARNING: This function demonstrates:
    1. Parameter estimation from historical data (μ, σ)
    2. Monte Carlo scenario generation
    3. Statistical aggregation (percentiles)
    4. Time path generation for visualization
    
    Args:
        df: Historical price data (used to estimate μ and σ)
        initial_value: Starting portfolio value (e.g., $100,000)
        time_horizon_years: How far into future to project (e.g., 1 year)
        n_scenarios: Number of random scenarios to generate (more = more accurate)
    
    Returns:
        dict with scenarios, percentiles, expected_return, volatility, time paths
    """
    # STEP 1: DATA LOADING
    # ---------------------
    if df is None:
        df = load_finance_data()

    # STEP 2: PARAMETER ESTIMATION
    # -----------------------------
    # LEARNING: Estimate μ (expected return) and σ (volatility) from historical data
    # These are the key parameters for the Geometric Brownian Motion model
    
    if df is None or len(df) == 0:
        # Default parameters if no data available
        # LEARNING: These are reasonable defaults for stock market:
        # - μ = 8% annual return (historical average)
        # - σ = 15% annual volatility (moderate risk)
        mu, sigma = 0.08, 0.15
    else:
        # STEP 3: CALCULATE RETURNS FROM HISTORICAL DATA
        # -----------------------------------------------
        # LEARNING: Sort by ticker and date for proper time series processing
        df = df.sort_values(['Ticker', 'Date'])
        
        # LEARNING: Calculate daily returns (percentage change)
        # This is the raw data we use to estimate μ and σ
        df['returns'] = df.groupby('Ticker')['Close'].pct_change()
        
        # LEARNING: Aggregate all returns across all tickers
        # This gives us a large sample to estimate population parameters
        all_returns = df['returns'].dropna()
        
        if len(all_returns) > 0:
            # STEP 4: ESTIMATE EXPECTED RETURN (μ)
            # -------------------------------------
            # LEARNING: Sample mean estimates population mean
            # Daily mean return → Annualized: μ_annual = μ_daily × 252
            mu = all_returns.mean() * 252
            
            # STEP 5: ESTIMATE VOLATILITY (σ)
            # --------------------------------
            # LEARNING: Sample standard deviation estimates population σ
            # Daily volatility → Annualized: σ_annual = σ_daily × √252
            # Why √252? Variance scales linearly, but std dev scales with √time
            sigma = all_returns.std() * np.sqrt(252)
        else:
            # Fallback to defaults if no valid returns
            mu, sigma = 0.08, 0.15

    # STEP 6: MONTE CARLO SIMULATION
    # -------------------------------
    # LEARNING: Generate n_scenarios random future outcomes
    # Each scenario is one possible path the portfolio could take
    
    dt = time_horizon_years  # Time step (1 year in this case)
    scenarios = []
    
    # LEARNING: Set random seed for reproducibility
    # Same seed = same random numbers = reproducible results
    np.random.seed(42)
    
    for _ in range(n_scenarios):
        # STEP 7: GENERATE RANDOM SHOCK
        # -----------------------------
        # LEARNING: Z ~ N(0,1) is a random variable from standard normal distribution
        # This represents the random component of price movement
        Z = np.random.randn()  # Sample from N(0,1)
        
        # STEP 8: APPLY GEOMETRIC BROWNIAN MOTION FORMULA
        # ------------------------------------------------
        # LEARNING: GBM formula: S(t) = S₀ × exp((μ - ½σ²)t + σ√t × Z)
        #
        # Breaking it down:
        # - (μ - ½σ²)t = drift term (expected growth, adjusted for log-normal)
        # - σ√t × Z = random shock (volatility × time × random factor)
        # - exp(...) = ensures positive prices (log-normal distribution)
        #
        # Why (μ - ½σ²)? This is the "drift adjustment" from Ito's Lemma
        value = initial_value * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        scenarios.append(float(value))

    # STEP 9: CALCULATE PERCENTILES
    # ------------------------------
    # LEARNING: Percentiles give us confidence intervals
    # After generating 1000 scenarios, we sort them and find key percentiles
    
    percentiles = {
        'p5': float(np.percentile(scenarios, 5)),    # 5% of scenarios below this (pessimistic)
        'p25': float(np.percentile(scenarios, 25)),  # 25% below (lower quartile)
        'p50': float(np.percentile(scenarios, 50)),  # 50% below (median)
        'p75': float(np.percentile(scenarios, 75)),  # 75% below (upper quartile)
        'p95': float(np.percentile(scenarios, 95))    # 95% below (optimistic)
    }
    
    # LEARNING: These percentiles create "fan charts" showing uncertainty bands
    # p5-p95 range shows 90% confidence interval (90% of scenarios fall in this range)

    # STEP 10: GENERATE TIME PATHS FOR VISUALIZATION
    # -----------------------------------------------
    # LEARNING: Create smooth curves showing how value evolves over time
    # This creates the "fan chart" visualization with multiple percentile bands
    
    n_points = 50  # Number of time points for smooth curve
    time_points = np.linspace(0, time_horizon_years, n_points)  # Evenly spaced from 0 to horizon
    
    # Initialize lists for each percentile path
    mean_path, p5_path, p25_path, p75_path, p95_path = [], [], [], [], []

    for t in time_points:
        # STEP 11: CALCULATE EXPECTED VALUE (MEAN PATH)
        # ----------------------------------------------
        # LEARNING: Expected value at time t: E[S(t)] = S₀ × exp(μt)
        # This is the "most likely" path (though individual scenarios will vary)
        mean_val = initial_value * np.exp((mu - 0.5 * sigma ** 2) * t)
        mean_path.append(float(mean_val))

        # STEP 12: CALCULATE PERCENTILE PATHS
        # ------------------------------------
        # LEARNING: Use inverse CDF (percentile point function) to find percentile values
        # norm.ppf(p) returns the value where p% of distribution is below it
        
        z5 = norm.ppf(0.05)   # Z-score for 5th percentile
        z25 = norm.ppf(0.25)  # Z-score for 25th percentile
        z75 = norm.ppf(0.75)  # Z-score for 75th percentile
        z95 = norm.ppf(0.95)  # Z-score for 95th percentile

        # LEARNING: Apply GBM formula with specific Z-scores for each percentile
        # This creates the uncertainty bands in the fan chart
        p5_path.append(float(initial_value * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * z5)))
        p25_path.append(float(initial_value * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * z25)))
        p75_path.append(float(initial_value * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * z75)))
        p95_path.append(float(initial_value * np.exp((mu - 0.5 * sigma ** 2) * t + sigma * np.sqrt(t) * z95)))

    # LEARNING: The result shows:
    # - Scenarios: 1000 individual possible outcomes
    # - Percentiles: Summary statistics (p5, p25, p50, p75, p95)
    # - Time paths: How each percentile evolves over time (for fan chart visualization)
    
    return {
        'scenarios': scenarios,
        'percentiles': percentiles,
        'expected_return': float(mu),
        'volatility': float(sigma),
        'time_points': time_points.tolist(),
        'mean_path': mean_path,
        'p5_path': p5_path,
        'p25_path': p25_path,
        'p75_path': p75_path,
        'p95_path': p95_path
    }
