"""
Game Theory Models for Finance
==============================

LEARNING: Game Theory in Financial Markets
------------------------------------------

This module demonstrates how game theory concepts apply to finance, revealing
strategic interactions between market participants.

THEORY: What is Game Theory?
----------------------------
Game theory studies strategic decision-making when outcomes depend on:
- Your actions
- Others' actions
- How actions interact

Key Concepts:
1. Players: Market participants (investors, traders, institutions)
2. Strategies: Available actions (buy, sell, hold, diversify)
3. Payoffs: Outcomes (returns, risk-adjusted returns)
4. Equilibrium: Stable outcome where no player wants to change strategy

WHY GAME THEORY IN FINANCE?
---------------------------
Financial markets are games:
- Multiple players (investors) making decisions
- Actions affect others (your trade moves prices)
- Strategic interactions (if others buy, should you buy too?)
- Competition and cooperation (diversification vs concentration)

APPLICATIONS:
-------------
1. Portfolio Optimization (Nash Equilibrium):
   - Each asset "competes" for portfolio weight
   - Equilibrium: No asset can improve by changing weight unilaterally
   - Result: Optimal diversification

2. Shapley Value (Fair Contribution):
   - Measures each asset's contribution to portfolio
   - Fair allocation of portfolio value
   - Used in attribution analysis

3. Prisoner's Dilemma (Cooperation vs Competition):
   - Cooperate = diversify (lower risk, lower return)
   - Defect = concentrate (higher risk, higher return)
   - Shows trade-off between diversification and concentration

4. Auction Theory (Asset Pricing):
   - Models how assets are priced in auctions (IPOs, bond auctions)
   - First-price: Winner pays their bid
   - Second-price (Vickrey): Winner pays second-highest bid
   - Helps understand market pricing mechanisms

5. Evolutionary Game Theory (Market Dynamics):
   - Strategies evolve over time (successful strategies spread)
   - Models how market behavior changes
   - Explains market trends and bubbles

MATHEMATICAL FOUNDATIONS:
-------------------------

1. NASH EQUILIBRIUM:
   -----------------
   Definition: Strategy profile where no player can improve by unilaterally changing strategy.
   
   Mathematical: For each player i:
      u_i(s_i*, s_{-i}*) ≥ u_i(s_i, s_{-i}*) for all s_i
   
   Where:
   - s_i* = Equilibrium strategy for player i
   - s_{-i}* = Equilibrium strategies for all other players
   - u_i = Utility function for player i
   
   LEARNING: In Nash equilibrium, everyone is doing their best given what others are doing.

2. SHAPLEY VALUE:
   -------------
   Formula: φ_i(v) = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] × [v(S ∪ {i}) - v(S)]
   
   Where:
   - φ_i = Shapley value for player i
   - v(S) = Value of coalition S
   - n = Total number of players
   - S = Coalition (subset of players)
   
   Interpretation: Average marginal contribution across all possible coalitions.
   
   LEARNING: Shapley value is the only fair allocation satisfying:
   - Efficiency: Sum of Shapley values = total value
   - Symmetry: Equal players get equal values
   - Dummy: Players who contribute nothing get zero
   - Additivity: Shapley value is additive

3. PRISONER'S DILEMMA:
   -------------------
   Payoff Matrix:
                Player 2
              Cooperate  Defect
   Player 1 C   (R, R)   (S, T)
            D   (T, S)   (P, P)
   
   Where: T > R > P > S (Temptation > Reward > Punishment > Sucker)
   
   Dominant Strategy: Defect (always better regardless of opponent's choice)
   Nash Equilibrium: (Defect, Defect) - both defect
   Paradox: Both would be better off cooperating, but rational play leads to defection!
   
   LEARNING: Shows why cooperation is difficult even when beneficial.

4. AUCTION THEORY:
   ---------------
   First-Price Sealed Bid:
   - Each bidder submits sealed bid
   - Highest bidder wins, pays their bid
   - Optimal strategy: Bid below true value (shade bid)
   
   Second-Price (Vickrey):
   - Highest bidder wins, pays second-highest bid
   - Optimal strategy: Bid true value (truthful bidding)
   
   LEARNING: Second-price auctions encourage truthful bidding!

5. EVOLUTIONARY STABLE STRATEGY (ESS):
   -----------------------------------
   Definition: Strategy that, if adopted by population, cannot be invaded by alternative strategy.
   
   Mathematical: For strategy s* to be ESS:
      u(s*, s*) > u(s, s*) for all s ≠ s*
      OR
      u(s*, s*) = u(s, s*) AND u(s*, s) > u(s, s) for all s ≠ s*
   
   LEARNING: ESS explains why certain strategies persist in markets.

LEARNING CHECKPOINT:
-------------------
1. What is Nash equilibrium?
   → Strategy profile where no player wants to change unilaterally

2. Why is Shapley value "fair"?
   → It's the only allocation satisfying efficiency, symmetry, dummy, and additivity

3. Why do both players defect in Prisoner's Dilemma?
   → Defect is dominant strategy (always better regardless of opponent)

Used by dashboards:
- Portfolio Optimization (Nash Equilibrium)
- Market Making Strategies
- Auction Theory (IPO/Bond Pricing)
- Prisoner's Dilemma (Cooperation Analysis)
- Shapley Value (Portfolio Contribution)
- Evolutionary Game Theory (Market Dynamics)

Algorithms implemented:
- Nash Equilibrium (iterative best response)
- Shapley Value (coalitional game theory)
- Prisoner's Dilemma payoff matrix
- Auction theory (first-price, second-price)
- Evolutionary stable strategies
"""

import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


def _load_finance_data():
    """Load finance data from standard paths."""
    import os
    # Get base directory (project root) - go up from ml/ to project root
    current_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(os.path.dirname(current_file))
    
    data_paths = [
        os.path.join(base_dir, 'data', 'raw', 'finance', 'risk_dashboard.csv'),
        os.path.join(base_dir, 'data', 'raw', 'finance', 'market_data.csv'),
        os.path.join(base_dir, 'data', 'raw', 'finance', 'market_data_real.csv'),
        os.path.join(base_dir, 'uploads', 'finance', 'market_data.csv'),
        os.path.join(base_dir, 'uploads', 'finance', 'market_data_real.csv')
    ]
    
    uploads_dir = os.path.join(base_dir, 'uploads', 'finance')
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            if filename.endswith('.csv'):
                data_paths.insert(0, os.path.join(uploads_dir, filename))
    
    for path in data_paths:
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                if 'date' in df.columns and 'Date' not in df.columns:
                    df['Date'] = df['date']
                if 'ticker' in df.columns and 'Ticker' not in df.columns:
                    df['Ticker'] = df['ticker']
                if 'Symbol' in df.columns and 'Ticker' not in df.columns:
                    df['Ticker'] = df['Symbol']
                if 'close' in df.columns and 'Close' not in df.columns:
                    df['Close'] = df['close']
                if 'price' in df.columns and 'Close' not in df.columns:
                    df['Close'] = df['price']
                
                if 'Date' in df.columns and ('Ticker' in df.columns or 'Symbol' in df.columns):
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    df = df.dropna(subset=['Date'])
                    if len(df) > 0:
                        return df
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue
    
    return None


def nash_equilibrium_portfolio(returns_matrix=None, risk_free_rate=0.02):
    """
    Compute Nash equilibrium portfolio weights using game theory.
    
    Each asset is a player, trying to maximize its contribution to portfolio utility.
    Nash equilibrium: no player can improve by unilaterally changing strategy.
    
    Args:
        returns_matrix: DataFrame with returns (rows=dates, cols=tickers) or None to load from file
        risk_free_rate: Risk-free rate (default 0.02)
    
    Returns:
        dict with keys:
            - nash_weights: dict of equilibrium weights
            - expected_return: portfolio expected return
            - volatility: portfolio volatility
            - sharpe_ratio: Sharpe ratio
            - convergence: whether algorithm converged
    """
    if returns_matrix is None:
        # Load data and compute returns
        df = _load_finance_data()
        if df is not None and not df.empty and 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            df = df.sort_values(['Ticker', 'Date'])
            df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
            returns_matrix = df_pivot.pct_change().dropna()
    
    if returns_matrix is None or len(returns_matrix) == 0:
        # Use default data
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        })
    
    # Calculate expected returns and covariance
    mu = returns_matrix.mean() * 252  # Annualized
    Sigma = returns_matrix.cov() * 252  # Annualized
    
    n_assets = len(mu)
    tickers = mu.index.tolist()
    
    # Nash equilibrium: iterative best response
    # Each player (asset) chooses weight to maximize utility given others' weights
    def utility(weights, mu_i, sigma_i, other_weights_sum, lambda_risk=2.0):
        """Utility function: return - risk_penalty"""
        portfolio_return = weights * mu_i + (1 - weights) * other_weights_sum
        portfolio_risk = weights**2 * sigma_i**2
        return portfolio_return - lambda_risk * portfolio_risk
    
    # Initialize weights
    weights = np.ones(n_assets) / n_assets
    max_iter = 100
    tolerance = 1e-6
    
    for iteration in range(max_iter):
        old_weights = weights.copy()
        
        # Each asset updates its weight (best response)
        for i in range(n_assets):
            mu_i = mu.iloc[i]
            sigma_i = np.sqrt(Sigma.iloc[i, i])
            other_weights_sum = (weights.sum() - weights[i]) / (n_assets - 1) if n_assets > 1 else 0
            
            # Find best response (weight that maximizes utility)
            def neg_utility(w):
                return -utility(w, mu_i, sigma_i, other_weights_sum)
            
            result = minimize(neg_utility, x0=weights[i], bounds=[(0, 1)], method='L-BFGS-B')
            if result.success:
                weights[i] = result.x[0]
        
        # Normalize weights
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(n_assets) / n_assets
        
        # Check convergence
        if np.linalg.norm(weights - old_weights) < tolerance:
            break
    
    # Calculate portfolio metrics
    portfolio_return = (weights * mu).sum()
    portfolio_variance = weights @ Sigma @ weights
    portfolio_vol = np.sqrt(portfolio_variance)
    sharpe = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
    
    return {
        'nash_weights': {ticker: float(w) for ticker, w in zip(tickers, weights)},
        'expected_return': float(portfolio_return),
        'volatility': float(portfolio_vol),
        'sharpe_ratio': float(sharpe),
        'convergence': iteration < max_iter - 1
    }


def shapley_value_portfolio(returns_matrix=None):
    """
    Compute Shapley value for each asset's contribution to portfolio.
    
    Shapley value: average marginal contribution across all possible coalitions.
    Measures fair contribution of each asset to portfolio performance.
    
    Args:
        returns_matrix: DataFrame with returns or None to load from file
    
    Returns:
        dict with keys:
            - shapley_values: dict of Shapley values per asset
            - marginal_contributions: list of marginal contributions
    """
    if returns_matrix is None:
        df = _load_finance_data()
        if df is not None and not df.empty and 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            df = df.sort_values(['Ticker', 'Date'])
            df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
            returns_matrix = df_pivot.pct_change().dropna()
    
    if returns_matrix is None or len(returns_matrix) == 0:
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        })
    
    mu = returns_matrix.mean() * 252
    Sigma = returns_matrix.cov() * 252
    
    n_assets = len(mu)
    tickers = mu.index.tolist()
    
    # Characteristic function: portfolio Sharpe ratio for coalition S
    def characteristic_function(S):
        """Returns portfolio Sharpe for coalition S (subset of assets)"""
        if len(S) == 0:
            return 0.0
        
        S_weights = np.ones(len(S)) / len(S)
        S_mu = mu[S].values
        S_Sigma = Sigma.loc[S, S].values
        
        portfolio_return = (S_weights @ S_mu)
        portfolio_variance = S_weights @ S_Sigma @ S_weights
        portfolio_vol = np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0.01
        
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        return sharpe
    
    from itertools import combinations
    shapley_values = {}
    
    for ticker in tickers:
        shapley = 0.0
        other_assets = [t for t in tickers if t != ticker]
        
        for S_size in range(len(other_assets) + 1):
            for S in combinations(other_assets, S_size):
                coalition = list(S)
                
                marginal = characteristic_function(coalition + [ticker]) - characteristic_function(coalition)
                
                weight = (math.factorial(len(coalition)) * math.factorial(n_assets - len(coalition) - 1)) / math.factorial(n_assets)
                shapley += weight * marginal
        
        shapley_values[ticker] = float(shapley)
    
    # Normalize to sum to total portfolio value
    total = sum(shapley_values.values())
    if total > 0:
        shapley_values = {k: v / total for k, v in shapley_values.items()}
    
    return {
        'shapley_values': shapley_values,
        'total_portfolio_value': float(total)
    }


def prisoner_dilemma_analysis(returns_matrix=None):
    """
    Analyze cooperation vs competition using Prisoner's Dilemma framework.
    
    Two strategies: Cooperate (diversify) vs Defect (concentrate).
    Payoff matrix based on portfolio performance.
    
    Args:
        returns_matrix: DataFrame with returns or None to load from file
    
    Returns:
        dict with keys:
            - payoff_matrix: 2x2 payoff matrix
            - nash_equilibrium: dominant strategy
            - cooperation_payoff: payoff if both cooperate
            - defection_payoff: payoff if both defect
    """
    if returns_matrix is None:
        df = _load_finance_data()
        if df is not None and not df.empty and 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            df = df.sort_values(['Ticker', 'Date'])
            df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
            returns_matrix = df_pivot.pct_change().dropna()
    
    if returns_matrix is None or len(returns_matrix) == 0:
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100)
        })
    
    mu = returns_matrix.mean() * 252
    Sigma = returns_matrix.cov() * 252
    
    # Strategy 1: Cooperate (diversified portfolio, equal weights)
    cooperate_weights = np.ones(len(mu)) / len(mu)
    cooperate_return = (cooperate_weights @ mu.values)
    cooperate_risk = np.sqrt(cooperate_weights @ Sigma.values @ cooperate_weights)
    cooperate_payoff = cooperate_return / cooperate_risk if cooperate_risk > 0 else 0
    
    # Strategy 2: Defect (concentrated, best asset only)
    best_asset_idx = mu.argmax()
    defect_weights = np.zeros(len(mu))
    defect_weights[best_asset_idx] = 1.0
    defect_return = mu.iloc[best_asset_idx]
    defect_risk = np.sqrt(Sigma.iloc[best_asset_idx, best_asset_idx])
    defect_payoff = defect_return / defect_risk if defect_risk > 0 else 0
    
    # Mixed strategy (one cooperates, one defects)
    mixed_payoff_coop = (cooperate_payoff + defect_payoff) / 2
    mixed_payoff_defect = defect_payoff  # Defector gets better payoff
    
    # Payoff matrix: [Cooperate, Defect] x [Cooperate, Defect]
    payoff_matrix = {
        'cooperate_cooperate': float(cooperate_payoff),
        'cooperate_defect': float(mixed_payoff_coop),
        'defect_cooperate': float(mixed_payoff_defect),
        'defect_defect': float(defect_payoff)
    }
    
    # Nash equilibrium: Defect is dominant strategy
    nash_equilibrium = 'defect_defect' if defect_payoff > cooperate_payoff else 'cooperate_cooperate'
    
    return {
        'payoff_matrix': payoff_matrix,
        'nash_equilibrium': nash_equilibrium,
        'cooperation_payoff': float(cooperate_payoff),
        'defection_payoff': float(defect_payoff),
        'temptation_payoff': float(mixed_payoff_defect),
        'sucker_payoff': float(mixed_payoff_coop)
    }


def auction_theory_pricing(returns_matrix=None, auction_type='first_price'):
    """
    Apply auction theory to asset pricing (IPO, bond auctions).
    
    First-price sealed bid: highest bidder wins, pays their bid.
    Second-price (Vickrey): highest bidder wins, pays second-highest bid.
    
    Args:
        returns_matrix: DataFrame with returns or None to load from file
        auction_type: 'first_price' or 'second_price'
    
    Returns:
        dict with keys:
            - optimal_bid: optimal bidding strategy
            - expected_price: expected auction price
            - winner_curse: winner's curse measure
    """
    if returns_matrix is None:
        df = _load_finance_data()
        if df is not None and not df.empty and 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            df = df.sort_values(['Ticker', 'Date'])
            df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
            returns_matrix = df_pivot.pct_change().dropna()
    
    if returns_matrix is None or len(returns_matrix) == 0:
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        })
    
    mu = returns_matrix.mean() * 252
    Sigma = returns_matrix.cov() * 252
    
    # True value = expected return
    true_values = mu.values
    
    # Bidders have private signals (noisy estimates of true value)
    n_bidders = len(mu)
    signals = true_values + np.random.normal(0, 0.1 * np.abs(true_values), n_bidders)
    
    # Optimal bidding strategy
    if auction_type == 'first_price':
        # First-price: bid below signal to account for winner's curse
        # Optimal bid = signal * (n-1)/n (risk-neutral)
        optimal_bids = signals * (n_bidders - 1) / n_bidders
    else:
        # Second-price: bid truthfully (signal = true value)
        optimal_bids = signals
    
    # Simulate auction
    winner_idx = np.argmax(optimal_bids)
    winning_bid = optimal_bids[winner_idx]
    
    if auction_type == 'first_price':
        price = winning_bid
    else:
        # Second-price: pay second-highest bid
        sorted_bids = np.sort(optimal_bids)
        price = sorted_bids[-2] if len(sorted_bids) > 1 else winning_bid
    
    # Winner's curse: overpayment relative to true value
    true_value = true_values[winner_idx]
    winner_curse = (price - true_value) / true_value if true_value > 0 else 0
    
    return {
        'optimal_bids': {ticker: float(bid) for ticker, bid in zip(mu.index, optimal_bids)},
        'winning_bid': float(winning_bid),
        'auction_price': float(price),
        'true_value': float(true_value),
        'winner_curse': float(winner_curse),
        'auction_type': auction_type
    }


def evolutionary_game_dynamics(returns_matrix=None, generations=50):
    """
    Simulate evolutionary game theory for market dynamics.
    
    Strategies evolve based on fitness (returns). More successful strategies
    increase in population share.
    
    Args:
        returns_matrix: DataFrame with returns or None to load from file
        generations: number of generations to simulate
    
    Returns:
        dict with keys:
            - strategy_shares: evolution of strategy shares over time
            - final_shares: final strategy distribution
            - evolution_stable: whether evolutionarily stable strategy reached
    """
    if returns_matrix is None:
        df = _load_finance_data()
        if df is not None and not df.empty and 'Date' in df.columns and 'Ticker' in df.columns and 'Close' in df.columns:
            df = df.sort_values(['Ticker', 'Date'])
            df_pivot = df.pivot(index='Date', columns='Ticker', values='Close')
            returns_matrix = df_pivot.pct_change().dropna()
    
    if returns_matrix is None or len(returns_matrix) == 0:
        returns_matrix = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
            'MSFT': np.random.normal(0.0008, 0.018, 100),
            'GOOGL': np.random.normal(0.0012, 0.022, 100)
        })
    
    mu = returns_matrix.mean() * 252
    n_strategies = len(mu)
    tickers = mu.index.tolist()
    
    # Initial strategy shares (equal)
    shares = np.ones(n_strategies) / n_strategies
    
    # Fitness = expected return (normalized)
    fitness = mu.values
    fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min() + 1e-10)
    
    # Evolution: replicator dynamics
    # d(share_i)/dt = share_i * (fitness_i - average_fitness)
    evolution_history = [shares.copy()]
    
    dt = 0.1  # Time step
    
    for gen in range(generations):
        avg_fitness = shares @ fitness
        
        # Replicator dynamics
        dshares = shares * (fitness - avg_fitness) * dt
        shares = shares + dshares
        
        # Normalize (keep on simplex)
        shares = np.maximum(shares, 0)  # No negative shares
        shares = shares / shares.sum() if shares.sum() > 0 else np.ones(n_strategies) / n_strategies
        
        evolution_history.append(shares.copy())
    
    # Check for evolutionarily stable strategy (ESS)
    # ESS: strategy that cannot be invaded by small fraction of mutants
    final_shares = shares
    dominant_strategy_idx = np.argmax(final_shares)
    is_ess = final_shares[dominant_strategy_idx] > 0.9  # Dominant strategy
    
    return {
        'strategy_shares': [[float(s) for s in shares] for shares in evolution_history],
        'final_shares': {ticker: float(s) for ticker, s in zip(tickers, final_shares)},
        'evolution_stable': bool(is_ess),
        'dominant_strategy': tickers[dominant_strategy_idx] if is_ess else None,
        'generations': generations
    }

