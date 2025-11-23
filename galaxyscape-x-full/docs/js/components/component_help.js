/**
 * Component Help System
 * Provides technical explanations for each chart/graph/KPI component
 * Includes theory, algorithms, equations, and analytical interpretation
 */

class ComponentHelp {
    constructor() {
        this.explanations = this.loadExplanations();
    }

    /**
     * Load component explanations with theory, algorithms, and equations
     */
    loadExplanations() {
        return {
            // Finance Components
            'finance': {
                'kpi': {
                    'risk-score': {
                        title: 'Portfolio Risk Score',
                        purpose: 'Quantifies overall portfolio risk on a 0-100 scale, combining multiple risk factors into a single metric using XGBoost/LightGBM models.',
                        theory: 'Risk scoring aggregates multiple risk dimensions (volatility, correlation, concentration, etc.) into a unified measure. Higher scores indicate greater potential for losses. ML models learn risk patterns from historical data.',
                        algorithm: 'XGBoost/LightGBM gradient boosting: Ensemble of decision trees trained on risk features (volatility, drawdown, correlation, concentration). Prediction: Risk Score = model.predict(features). Calibration: Score normalized to 0-100 range.',
                        equation: 'ML model: $\\text{Risk Score} = f_{\\text{model}}(\\sigma, \\text{MDD}, \\rho, C, ...)$ where $f$ = trained XGBoost/LightGBM function. Calibration: $\\text{Score}_{\\text{norm}} = \\frac{\\text{Score} - \\min}{\\max - \\min} \\times 100$',
                        interpretation: '0-30: Low Risk (Conservative), 31-60: Moderate Risk (Balanced), 61-80: High Risk (Aggressive), 81-100: Critical Risk (Very Aggressive). Model learns non-linear risk relationships.',
                        analysis: 'Monitor trends over time. Sudden increases may indicate market stress or portfolio concentration issues. Compare to historical risk scores for context.'
                    },
                    'var': {
                        title: 'Value at Risk (VaR)',
                        purpose: 'Estimates maximum potential loss at a given confidence level over a specified time horizon.',
                        theory: 'VaR represents the worst-case loss that should not be exceeded with a certain probability (e.g., 95% confidence means 5% chance of exceeding this loss).',
                        algorithm: 'Historical Simulation or Parametric Method using normal distribution assumption',
                        equation: '$$\\text{VaR}(\\alpha) = \\mu - z_{\\alpha} \\times \\sigma \\times \\sqrt{T}$$ where $\\mu$ = mean return, $\\sigma$ = volatility, $T$ = time horizon, $z_{\\alpha}$ = quantile (1.645 for 95%)',
                        interpretation: 'VaR(95%) = $10,000 means there is a 5% chance of losing more than $10,000 over the time period. Lower VaR indicates lower risk.',
                        analysis: 'Compare VaR to actual losses. If losses frequently exceed VaR, the model may underestimate risk. Use for position sizing and risk limits.'
                    },
                    'volatility': {
                        title: 'Volatility',
                        purpose: 'Measures the degree of variation in asset prices over time, indicating price uncertainty. Can be calculated from historical data or predicted using LSTM models.',
                        theory: 'Volatility (σ) is the standard deviation of returns, representing how much prices fluctuate. Higher volatility means greater price uncertainty and risk. LSTM models can forecast future volatility.',
                        algorithm: 'Historical: σ = std(returns) × √252 (annualized). LSTM: Neural network with LSTM layers learns temporal patterns in volatility. Returns: r_t = (P_t - P_{t-1}) / P_{t-1}',
                        equation: 'Historical: $\\sigma = \\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} (r_i - \\bar{r})^2} \\times \\sqrt{252}$ where $r_i$ = daily returns, $\\bar{r}$ = mean return. LSTM: $\\sigma_t = \\text{LSTM}(r_{t-w:t-1})$ where $w$ = window size, LSTM uses hidden states $h_t = \\tanh(W_h h_{t-1} + W_x r_t + b)$',
                        interpretation: 'Low volatility (<15%): Stable prices, Low Risk. Medium (15-30%): Moderate fluctuations. High (>30%): Large price swings, High Risk. LSTM predictions enable forward-looking risk assessment.',
                        analysis: 'Volatility clustering: High volatility periods tend to follow high volatility. Use for risk assessment and option pricing. Compare historical vs LSTM-predicted volatility for forecasting accuracy.'
                    },
                    'projected-value': {
                        title: 'Projected Portfolio Value (Monte Carlo)',
                        purpose: 'Mean projected future portfolio value from Monte Carlo simulation over specified time horizon.',
                        theory: 'Monte Carlo simulation generates probability distribution of future values by sampling from historical return distributions. Mean represents expected outcome.',
                        algorithm: 'Monte Carlo: Generate S simulations, each with T daily returns sampled from N(μ, σ²). Portfolio value: V_t = V_0 × Π(1 + r_i). Mean: E[V_T] = (1/S) × Σ V_T^(s)',
                        equation: '$$E[V_T] = \\frac{1}{S} \\sum_{s=1}^{S} V_T^{(s)}$$ where $V_T^{(s)} = V_0 \\prod_{i=1}^{T} (1 + r_i^{(s)})$, $r_i^{(s)} \\sim N(\\mu, \\sigma^2)$, $S$ = simulations, $T$ = trading days',
                        interpretation: 'Mean projected value = expected outcome. Compare to current value to assess expected growth. Higher mean = better expected performance.',
                        analysis: 'Use mean for planning. Consider percentiles (5th, 95th) for risk assessment. Compare to current value to estimate expected return. Validate against historical performance.'
                    },
                    'expected-return': {
                        title: 'Expected Return %',
                        purpose: 'Expected annualized return percentage from Monte Carlo simulation, calculated as (mean_projected / current_value - 1) × 100%',
                        theory: 'Expected return quantifies anticipated portfolio growth rate. Based on historical return distribution and Monte Carlo simulation results.',
                        algorithm: 'Expected return: ER = (E[V_T] / V_0 - 1) / time_horizon × 100%. Alternatively: ER = mean(historical_returns) × 252 × 100% (annualized)',
                        equation: '$$\\text{ER} = \\frac{E[V_T] - V_0}{V_0 \\times T} \\times 100\\%$$ or $$\\text{ER} = \\bar{r} \\times 252 \\times 100\\%$$ where $\\bar{r}$ = mean daily return, 252 = trading days/year',
                        interpretation: 'Positive ER = expected growth. Negative = expected decline. Higher ER = better expected performance. Compare to risk-free rate (e.g., Treasury yield) for risk-adjusted assessment.',
                        analysis: 'Use for return expectations. Compare to historical returns for validation. Consider Sharpe ratio (ER / volatility) for risk-adjusted performance. Higher ER with acceptable risk = better portfolio.'
                    },
                    'nash-sharpe': {
                        title: 'Nash Equilibrium Sharpe Ratio',
                        purpose: 'Portfolio Sharpe ratio at Nash equilibrium, where no asset can improve by unilaterally changing its weight.',
                        theory: 'Nash equilibrium in portfolio optimization: each asset (player) chooses weight to maximize utility given others\' weights. Equilibrium: no player can improve by changing strategy alone. Represents stable portfolio allocation.',
                        algorithm: 'Iterative best response: For each asset i, find weight w_i that maximizes utility U(w_i, w_{-i}) given other weights. Repeat until convergence. Utility: U = return - λ × risk².',
                        equation: '$$w_i^* = \\arg\\max_{w_i} U(w_i, w_{-i}) = \\arg\\max_{w_i} [w_i \\mu_i + (1-w_i)\\bar{\\mu}_{-i} - \\lambda w_i^2 \\sigma_i^2]$$ where $\\mu_i$ = expected return, $\\sigma_i$ = volatility, $\\lambda$ = risk aversion. Equilibrium: $w^*$ such that $w_i^*$ is best response to $w_{-i}^*$ for all $i$.',
                        interpretation: 'Higher Sharpe = better risk-adjusted return at equilibrium. Nash equilibrium represents stable allocation where no asset wants to change weight. Compare to equal-weight or market-cap weights.',
                        analysis: 'Use for portfolio optimization. Nash equilibrium provides game-theoretic foundation for asset allocation. Higher Sharpe indicates better equilibrium. Monitor convergence to ensure stable solution.'
                    },
                    'nash-return': {
                        title: 'Nash Equilibrium Expected Return',
                        purpose: 'Expected portfolio return at Nash equilibrium allocation.',
                        theory: 'Portfolio expected return at Nash equilibrium weights. Each asset contributes according to its equilibrium weight.',
                        algorithm: 'Portfolio return: $\\mu_p = \\sum_i w_i^* \\mu_i$ where $w_i^*$ = Nash equilibrium weights, $\\mu_i$ = asset expected returns.',
                        equation: '$$\\mu_p = \\sum_{i=1}^{n} w_i^* \\mu_i$$ where $w_i^*$ = Nash equilibrium weight for asset $i$, $\\mu_i$ = expected return of asset $i$.',
                        interpretation: 'Higher return = better expected performance. Compare to risk-free rate and market return. Risk-adjusted return (Sharpe) more important than raw return.',
                        analysis: 'Use for return expectations. Compare to historical returns for validation. Consider risk (volatility) for risk-adjusted assessment.'
                    },
                    'nash-volatility': {
                        title: 'Nash Equilibrium Portfolio Volatility',
                        purpose: 'Portfolio volatility at Nash equilibrium weights, measuring risk.',
                        theory: 'Portfolio volatility at Nash equilibrium. Accounts for diversification benefits and correlations.',
                        algorithm: 'Portfolio variance: $\\sigma_p^2 = w^* \\Sigma w^{*T}$ where $w^*$ = Nash weights, $\\Sigma$ = covariance matrix. Volatility: $\\sigma_p = \\sqrt{\\sigma_p^2}$.',
                        equation: '$$\\sigma_p^2 = \\sum_{i=1}^{n} \\sum_{j=1}^{n} w_i^* w_j^* \\sigma_{ij}$$ where $\\sigma_{ij}$ = covariance between assets $i$ and $j$. Volatility: $\\sigma_p = \\sqrt{\\sigma_p^2}$.',
                        interpretation: 'Lower volatility = lower risk. Diversification reduces volatility through negative correlations. Higher volatility = higher uncertainty.',
                        analysis: 'Use for risk assessment. Compare to individual asset volatilities to assess diversification benefits. Lower volatility with similar return = better portfolio.'
                    },
                    'default': {
                        title: 'KPI Metric',
                        purpose: 'Key Performance Indicator showing a critical metric for decision-making.',
                        theory: 'KPIs provide quantifiable measures of performance or risk, enabling data-driven decisions.',
                        algorithm: 'Calculation method depends on specific metric type',
                        equation: 'Varies by metric',
                        interpretation: 'Interpret values relative to benchmarks, historical averages, or target thresholds.',
                        analysis: 'Track trends over time and compare to industry standards or portfolio objectives.'
                    }
                },
                'gauge': {
                    'risk-level': {
                        title: 'Risk Level Gauge',
                        purpose: 'Visual indicator of current portfolio risk status using color-coded gauge based on ML-predicted risk score.',
                        theory: 'Gauges provide immediate visual feedback on risk status, using color psychology (green=safe, yellow=caution, red=danger). Risk score from XGBoost/LightGBM models.',
                        algorithm: 'Risk level determined by risk score thresholds: Low (<30), Medium (30-60), High (60-80), Critical (>80). Risk score = ML_model.predict(features) normalized to 0-100',
                        equation: '$$\\text{Risk Level} = \\begin{cases} \\text{Low} & \\text{if Score} < 30 \\\\ \\text{Medium} & \\text{if } 30 \\leq \\text{Score} < 60 \\\\ \\text{High} & \\text{if } 60 \\leq \\text{Score} < 80 \\\\ \\text{Critical} & \\text{if Score} \\geq 80 \\end{cases}$$',
                        interpretation: 'Green: Safe to proceed. Yellow: Monitor closely. Orange: Consider reducing exposure. Red: Immediate action required. Based on ML model predictions.',
                        analysis: 'Use for quick risk assessment. Combine with detailed metrics for comprehensive analysis. Monitor changes over time to detect risk regime shifts.'
                    },
                    'volatility': {
                        title: 'Portfolio Volatility Gauge',
                        purpose: 'Shows portfolio volatility level as a gauge, indicating price uncertainty and risk.',
                        theory: 'Volatility measures price fluctuation. Higher volatility = higher risk. Can be historical (rolling window) or LSTM-predicted.',
                        algorithm: 'Volatility calculation: σ = std(returns) × √252 (annualized). Returns: r_t = (P_t - P_{t-1}) / P_{t-1}. Gauge value = σ normalized to 0-100 scale',
                        equation: '$$\\sigma = \\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} (r_i - \\bar{r})^2} \\times \\sqrt{252}$$ where $r_i$ = daily returns. Gauge: $\\text{Gauge} = \\frac{\\sigma - \\sigma_{\\min}}{\\sigma_{\\max} - \\sigma_{\\min}} \\times 100$',
                        interpretation: 'Low gauge (<30%): Low volatility, stable prices. Medium (30-60%): Moderate fluctuations. High (>60%): High volatility, large price swings.',
                        analysis: 'Monitor volatility trends. High volatility periods cluster together. Use for dynamic risk management and position sizing.'
                    },
                    'default': {
                        title: 'Gauge Chart',
                        purpose: 'Circular gauge visualization showing a metric value within a defined range.',
                        theory: 'Gauges provide intuitive visual representation of a value relative to min/max thresholds.',
                        algorithm: 'Arc rendering based on value position within range',
                        equation: '$$\\theta = \\frac{\\text{value} - \\text{min}}{\\text{max} - \\text{min}} \\times 360°$$',
                        interpretation: 'Value position indicates status relative to thresholds. Center typically represents optimal range.',
                        analysis: 'Monitor gauge position over time. Rapid changes may indicate significant shifts requiring attention.'
                    }
                },
                'line-chart': {
                    'risk-trends': {
                        title: 'Risk Trends Time Series',
                        purpose: 'Shows how portfolio risk evolves over time, identifying trends and patterns.',
                        theory: 'Time series analysis reveals risk dynamics, cyclical patterns, and trend changes. Helps predict future risk levels.',
                        algorithm: 'Moving average smoothing: MA(t) = (1/n) × Σ(x_{t-i}) for i=0 to n-1. Risk calculated as weighted combination of volatility, correlation, and concentration.',
                        equation: '$$\\text{Risk}(t) = 0.4 \\times \\sigma(t) + 0.3 \\times \\text{MDD}(t) + 0.2 \\times \\rho(t) + 0.1 \\times C(t)$$ where $\\sigma$ = volatility, MDD = maximum drawdown, $\\rho$ = correlation risk, $C$ = concentration',
                        interpretation: 'Upward trend: Increasing risk, consider reducing exposure. Downward trend: Decreasing risk, may allow increased positions. Flat: Stable risk profile.',
                        analysis: 'Identify regime changes (sudden shifts), seasonality (recurring patterns), and volatility clustering (high risk periods followed by high risk).'
                    },
                    'returns-comparison': {
                        title: 'Returns Comparison (1M, 3M, 1Y)',
                        purpose: 'Compares percentage returns across different time periods (1 month, 3 months, 1 year) for multiple stocks.',
                        theory: 'Returns measure price appreciation over time. Comparing returns across periods reveals performance consistency and trend direction.',
                        algorithm: 'Return calculation: r = (P_t / P_0 - 1) × 100% where P_t = current price, P_0 = price at period start. Annualized for 1Y: r_annual = ((P_t / P_0)^(252/days) - 1) × 100%',
                        equation: '$$\\text{Return}_{\\text{period}} = \\left(\\frac{P_{\\text{current}}}{P_{\\text{start}}} - 1\\right) \\times 100\\%$$ where $P_{\\text{current}}$ = current price, $P_{\\text{start}}$ = price at period start (1M, 3M, or 1Y ago)',
                        interpretation: 'Positive returns = price appreciation. Negative = depreciation. Higher returns = better performance. Compare across periods to assess consistency.',
                        analysis: 'Stocks with consistent positive returns across all periods show strong momentum. Diverging returns may indicate volatility or changing fundamentals. Use for stock selection and portfolio rebalancing.'
                    },
                    'volatility': {
                        title: 'Volatility Time Series',
                        purpose: 'Tracks volatility changes over time, showing periods of high and low market uncertainty.',
                        theory: 'Volatility is not constant; it varies over time (heteroscedasticity). High volatility periods cluster together.',
                        algorithm: 'Rolling window standard deviation: σ_t = std(returns[t-w:t]) where w = window size',
                        equation: '$$\\sigma_t = \\sqrt{\\frac{1}{w-1} \\sum_{i=0}^{w-1} (r_{t-i} - \\bar{r}_t)^2}$$ where $w$ = window size',
                        interpretation: 'Spikes indicate market stress or uncertainty. Sustained high volatility suggests ongoing market instability.',
                        analysis: 'Compare to historical averages. Volatility above 2× historical average may indicate crisis conditions. Use for dynamic risk management.'
                    },
                    'default': {
                        title: 'Line Chart',
                        purpose: 'Displays time series data showing how a metric changes over time.',
                        theory: 'Line charts reveal trends, patterns, and relationships in temporal data.',
                        algorithm: 'Linear interpolation between data points',
                        equation: '$$y(t) = f(x(t))$$ where $t$ = time, $x$ = input variable',
                        interpretation: 'Slope indicates rate of change. Steep slopes suggest rapid changes. Peaks/valleys indicate extremes.',
                        analysis: 'Identify trends (upward/downward), cycles (periodic patterns), and anomalies (unusual spikes or drops).'
                    }
                },
                'bar-chart': {
                    'price-comparison': {
                        title: 'Price Comparison Bar Chart',
                        purpose: 'Compares current prices across multiple stocks or assets side-by-side.',
                        theory: 'Bar charts enable direct visual comparison of values across categories, making relative differences immediately apparent.',
                        algorithm: 'Bar height proportional to value: height = (value / max_value) × chart_height',
                        equation: '$$\\text{Bar}_i = \\frac{\\text{Price}_i}{\\max(\\text{Prices})} \\times H_{\\max}$$',
                        interpretation: 'Taller bars indicate higher prices. Compare relative heights to assess price differences. Use for identifying outliers.',
                        analysis: 'Compare prices to historical averages or sector peers. Significant deviations may indicate over/under-valuation.'
                    },
                    'default': {
                        title: 'Bar Chart',
                        purpose: 'Compares values across different categories or groups.',
                        theory: 'Bar charts provide categorical comparison, making it easy to identify highest/lowest values.',
                        algorithm: 'Bar rendering based on value magnitude',
                        equation: '$$\\text{Bar height} \\propto \\text{value}$$',
                        interpretation: 'Relative bar heights show comparative magnitudes. Tallest bars represent highest values.',
                        analysis: 'Identify top/bottom performers, outliers, and distribution patterns across categories.'
                    }
                },
                'pie-chart': {
                    'sector-breakdown': {
                        title: 'Sector Breakdown Pie Chart',
                        purpose: 'Shows portfolio allocation across different market sectors.',
                        theory: 'Pie charts represent parts of a whole, showing proportional allocation. Each slice represents a sector\'s share.',
                        algorithm: 'Angle calculation: θ_i = (value_i / Σvalues) × 360°',
                        equation: '$$\\text{Sector\\%} = \\frac{\\text{Sector Value}}{\\text{Total Portfolio Value}} \\times 100\\%$$',
                        interpretation: 'Larger slices indicate greater allocation. Balanced portfolios show similar slice sizes. Concentrated portfolios show one or two large slices.',
                        analysis: 'Assess diversification: Many small slices = well-diversified. Few large slices = concentrated risk. Aim for balanced allocation across sectors.'
                    },
                    'default': {
                        title: 'Pie Chart',
                        purpose: 'Shows proportional distribution of a whole across categories.',
                        theory: 'Pie charts visualize part-to-whole relationships, making proportions immediately visible.',
                        algorithm: 'Circular sector rendering based on proportional values',
                        equation: '$$\\text{Slice\\%} = \\frac{\\text{Category Value}}{\\text{Total}} \\times 100\\%$$',
                        interpretation: 'Larger slices represent larger proportions. All slices sum to 100%.',
                        analysis: 'Identify dominant categories, balance, and outliers in the distribution.'
                    }
                },
                'network-graph': {
                    'correlation-network': {
                        title: 'Correlation Network Graph',
                        purpose: 'Visualizes relationships between assets using network topology, where nodes are assets and edges represent correlations above a threshold. Edge thickness varies with correlation strength.',
                        theory: 'Correlation network reveals hidden relationships and clusters. Strong correlations create dense subgraphs (sector clusters). Central nodes are highly connected (systemically important). Network structure helps identify diversification opportunities and contagion risks.',
                        algorithm: 'Compute correlation matrix: ρ_ij = Cov(r_i, r_j) / (σ_i × σ_j) from returns. Create edges for |ρ_ij| ≥ threshold. Node size ∝ volatility. Edge width ∝ |ρ_ij|. Force-directed layout positions nodes.',
                        equation: 'Correlation: $\\rho_{ij} = \\frac{\\text{Cov}(r_i, r_j)}{\\sigma_i \\times \\sigma_j}$ where $r_i, r_j$ = returns, $\\sigma$ = volatility. Edge creation: $E = \\{(i,j) : |\\rho_{ij}| \\geq \\theta\\}$ where $\\theta$ = threshold (default 0.5). Edge weight: $w_{ij} = |\\rho_{ij}|$.',
                        interpretation: 'Thick edges = strong correlation (|ρ| > 0.7). Thin edges = moderate (0.5 ≤ |ρ| < 0.7). Dense clusters = highly correlated groups (sector clusters, market segments). Central nodes = systemically important assets (high degree). Isolated nodes = uncorrelated assets (diversification opportunities).',
                        analysis: 'Identify diversification opportunities (weakly connected assets), contagion risks (dense clusters), and systemic importance (central nodes). Use for portfolio optimization: add uncorrelated assets to reduce risk. Monitor cluster changes over time for regime shifts.'
                    },
                    'default': {
                        title: 'Network Graph',
                        purpose: 'Displays relationships between entities as a network of nodes and edges.',
                        theory: 'Network analysis reveals structure, clusters, and central entities in complex systems.',
                        algorithm: 'Graph layout algorithms (force-directed, hierarchical, circular)',
                        equation: 'Graph G = (V, E) where V = vertices (nodes), E = edges (relationships)',
                        interpretation: 'Node size = importance. Edge thickness = relationship strength. Clusters = groups of related entities.',
                        analysis: 'Identify key entities (central nodes), communities (clusters), and relationship patterns (dense vs sparse regions).'
                    }
                },
                'data-table': {
                    'stock-table': {
                        title: 'Stock Data Table',
                        purpose: 'Tabular display of stock metrics enabling detailed comparison and filtering.',
                        theory: 'Tables provide precise numerical values and enable sorting/filtering for detailed analysis.',
                        algorithm: 'Tabular rendering with sortable columns and filterable rows. Metrics calculated from yfinance data: price, returns, market cap, P/E, beta, etc.',
                        equation: 'Various metrics: $\\text{P/E} = \\frac{\\text{Price}}{\\text{Earnings}}$, $\\beta = \\frac{\\text{Cov}(r_{\\text{stock}}, r_{\\text{market}})}{\\text{Var}(r_{\\text{market}})}$, $\\text{Return\\%} = \\frac{P_{\\text{current}} - P_{\\text{prev}}}{P_{\\text{prev}}} \\times 100\\%$',
                        interpretation: 'Sort by different columns to identify top/bottom performers. Filter to focus on specific criteria (sector, risk level, etc.).',
                        analysis: 'Compare metrics across stocks. Low P/E may indicate value. High beta indicates market sensitivity. Use for stock selection.'
                    },
                    'scenarios': {
                        title: 'Detailed Scenarios (Future Outcomes)',
                        purpose: 'Shows Monte Carlo simulation results for portfolio future value projections across different probability scenarios.',
                        theory: 'Monte Carlo simulation generates thousands of possible future outcomes by sampling from historical return distributions, providing probabilistic forecasts.',
                        algorithm: 'Monte Carlo simulation: For each simulation, generate random returns from normal distribution N(μ, σ²) based on historical data, then compound over time horizon. Calculate percentiles (5th, 25th, 50th, 75th, 95th) from distribution.',
                        equation: '$$V_t = V_0 \\prod_{i=1}^{T} (1 + r_i)$$ where $V_0$ = current value, $r_i \\sim N(\\mu, \\sigma^2)$ = random daily return, $T$ = trading days. Percentiles: $V_{p} = \\text{Percentile}(\\{V_t^{(s)}\\}_{s=1}^{S}, p)$ where $S$ = number of simulations',
                        interpretation: 'Conservative (5th percentile) = worst-case scenario. Moderate (50th percentile) = median outcome. Optimistic (95th percentile) = best-case scenario. Wider spread = higher uncertainty.',
                        analysis: 'Use scenarios for risk assessment and planning. If 5th percentile is below acceptable threshold, consider reducing risk. Compare scenarios to assess portfolio robustness.'
                    },
                    'default': {
                        title: 'Data Table',
                        purpose: 'Structured tabular display of data with rows and columns.',
                        theory: 'Tables provide detailed, precise data access for analysis and comparison.',
                        algorithm: 'Row/column rendering with sorting and filtering',
                        equation: 'N/A - Display format',
                        interpretation: 'Each row = one entity. Each column = one attribute. Sort/filter to find patterns.',
                        analysis: 'Use sorting to identify extremes. Use filtering to focus on subsets. Compare values across rows and columns.'
                    }
                },
                'leaderboard': {
                    'top-performers': {
                        title: 'Top Performers Leaderboard',
                        purpose: 'Ranks stocks by performance metric, showing best and worst performers.',
                        theory: 'Ranking enables quick identification of top/bottom performers for decision-making.',
                        algorithm: 'Sort by metric value: sorted(stocks, key=lambda x: x.metric, reverse=True)',
                        equation: '$$\\text{Rank} = \\text{position in sorted list by metric value}$$',
                        interpretation: 'Top ranks = best performers. Bottom ranks = worst performers. Use for stock selection and portfolio rebalancing.',
                        analysis: 'Identify consistent top performers (momentum), improving stocks (rising ranks), and declining stocks (falling ranks).'
                    },
                    'default': {
                        title: 'Leaderboard',
                        purpose: 'Ranked list showing top/bottom entities by a specific metric.',
                        theory: 'Rankings provide competitive comparison, highlighting best and worst performers.',
                        algorithm: 'Sorting algorithm (quicksort, mergesort) by metric value',
                        equation: '$$\\text{Rank}_i = \\text{position}(i, \\text{sorted}(\\text{entities}, \\text{by metric}))$$',
                        interpretation: 'Higher ranks = better performance. Lower ranks = worse performance.',
                        analysis: 'Track rank changes over time. Identify rising stars and declining entities.'
                    }
                }
            },
            // Astronomy Components
            'astronomy': {
                'kpi': {
                    'total-stars': {
                        title: 'Total Stars',
                        purpose: 'Count of stellar objects in the dataset, representing the sample size for analysis.',
                        theory: 'Sample size affects statistical significance. Larger samples provide more reliable statistics and better ML model performance.',
                        algorithm: 'Simple count: N = Σ(1 for each star)',
                        equation: '$$N = |S|$$ where $S$ = set of stars',
                        interpretation: 'Larger N = more reliable statistics. Small N (<100) may have high variance. Large N (>10,000) provides robust analysis.',
                        analysis: 'Use sample size to assess confidence in results. Larger datasets enable more sophisticated analysis and better generalization.'
                    },
                    'avg-age': {
                        title: 'Average Stellar Age',
                        purpose: 'Mean estimated age of stars in the dataset, indicating the typical stellar population age.',
                        theory: 'Stellar age relates to evolutionary stage. Older stars are typically redder and cooler. Age distribution reveals population characteristics.',
                        algorithm: 'Arithmetic mean: Agē = (1/N) × Σ(Age_i)',
                        equation: '$$\\mu_{\\text{age}} = \\frac{1}{n} \\sum_{i=1}^{n} \\text{Age}_i$$',
                        interpretation: 'High average age (>10 Gyr) = old population. Low average age (<1 Gyr) = young population. Compare to theoretical models.',
                        analysis: 'Compare to stellar evolution models. Deviations may indicate selection bias or interesting astrophysical phenomena.'
                    },
                    'model-mae': {
                        title: 'Model MAE (Mean Absolute Error)',
                        purpose: 'Measures average prediction error magnitude for XGBoost or LightGBM stellar age models.',
                        theory: 'MAE quantifies average absolute deviation between predicted and actual values. Lower MAE = better model. Robust to outliers compared to RMSE.',
                        algorithm: 'MAE calculation: MAE = (1/n) × Σ|y_i - ŷ_i| where y = actual, ŷ = predicted, n = number of samples. Computed after model training on test/validation set.',
                        equation: '$$\\text{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|$$ where $y_i$ = actual age, $\\hat{y}_i$ = predicted age, $n$ = number of samples',
                        interpretation: 'Lower MAE = better model accuracy. MAE in same units as target (e.g., years or Gyr). MAE = 0.5 Gyr means average error is 500 million years.',
                        analysis: 'Compare MAE across models (XGBoost vs LightGBM). Lower MAE indicates better predictions. Use for model selection and performance monitoring.'
                    },
                    'best-model': {
                        title: 'Best Model (Lowest MAE)',
                        purpose: 'Indicates which ML model (XGBoost or LightGBM) performs best based on MAE comparison.',
                        theory: 'Model comparison helps select optimal algorithm. Best model has lowest prediction error (MAE) on validation data.',
                        algorithm: 'Compare MAE values: best_model = argmin(MAE_xgboost, MAE_lightgbm). Both models trained on same data with same features.',
                        equation: '$$\\text{Best Model} = \\text{argmin}_{m \\in \\{\\text{XGBoost}, \\text{LightGBM}\\}} \\text{MAE}_m$$',
                        interpretation: 'XGBoost = XGBoost has lower MAE. LightGBM = LightGBM has lower MAE. Similar MAE = models perform comparably.',
                        analysis: 'Use best model for predictions. Consider other factors: training time, interpretability, feature importance. Both are gradient boosting algorithms with similar performance.'
                    },
                    'default': {
                        title: 'KPI Metric',
                        purpose: 'Key Performance Indicator for stellar analysis.',
                        theory: 'KPIs provide summary statistics for quick assessment of dataset characteristics.',
                        algorithm: 'Calculation depends on metric type',
                        equation: 'Varies by metric',
                        interpretation: 'Interpret relative to theoretical expectations or comparison datasets.',
                        analysis: 'Track trends and compare to astrophysical models or literature values.'
                    }
                },
                'line-chart': {
                    'age-distribution': {
                        title: 'Age Distribution Over Time',
                        purpose: 'Shows how stellar ages are distributed, revealing population characteristics and evolutionary stages.',
                        theory: 'Age distribution reflects star formation history. Peaks indicate star formation bursts. Smooth distribution suggests continuous formation.',
                        algorithm: 'Histogram or kernel density estimation',
                        equation: '$$\\text{PDF}(\\text{age}) = \\frac{1}{n} \\sum_{i=1}^{n} K\\left(\\frac{\\text{age} - \\text{Age}_i}{h}\\right)$$ where $K$ = kernel function, $h$ = bandwidth',
                        interpretation: 'Peaks = common ages (star formation events). Tails = rare ages. Shape reveals formation history.',
                        analysis: 'Compare to theoretical star formation rate models. Identify formation bursts and evolutionary stages.'
                    },
                    'light-curve': {
                        title: 'Light Curve (Flux vs Time)',
                        purpose: 'Shows how stellar brightness changes over time, revealing variability, transits, and stellar activity.',
                        theory: 'Light curves reveal stellar physics: rotation (periodic), transits (dips), flares (spikes), pulsations (oscillations).',
                        algorithm: 'Time series analysis with period detection: P = argmax(PSD(f)) where PSD = power spectral density',
                        equation: '$$\\text{Flux}(t) = F_0 + A \\sin\\left(\\frac{2\\pi t}{P} + \\phi\\right)$$ for periodic variations, where $F_0$ = baseline flux, $A$ = amplitude, $P$ = period, $\\phi$ = phase',
                        interpretation: 'Periodic = rotation or binary. Dips = transits or eclipses. Spikes = flares. Smooth = stable star.',
                        analysis: 'Detect periods using FFT or Lomb-Scargle. Identify transits (exoplanet candidates). Classify variability types.'
                    },
                    'regression-plot': {
                        title: 'Predicted vs Actual Age (Regression Plot)',
                        purpose: 'Shows model performance by comparing predicted stellar ages (from XGBoost/LightGBM) to actual ages.',
                        theory: 'Regression plots reveal model accuracy. Points on diagonal line (y=x) indicate perfect predictions. Deviations show prediction errors.',
                        algorithm: 'XGBoost/LightGBM gradient boosting: Ensemble of decision trees. Prediction: ŷ = Σ(tree_i(x)). Training: Minimize loss function L(y, ŷ) using gradient descent.',
                        equation: 'XGBoost objective: $L = \\sum_{i=1}^{n} l(y_i, \\hat{y}_i) + \\sum_{k=1}^{K} \\Omega(f_k)$ where $l$ = loss (MSE), $\\Omega$ = regularization. Prediction: $\\hat{y} = \\sum_{k=1}^{K} f_k(x)$ where $f_k$ = tree $k$',
                        interpretation: 'Points on y=x line = perfect predictions. Points above line = overestimation. Points below = underestimation. Tight scatter = good model. Wide scatter = poor model.',
                        analysis: 'Calculate R² (coefficient of determination) and correlation between predicted and actual. Identify systematic biases (over/under-estimation in certain ranges). Use for model validation and improvement.'
                    },
                    'residuals-plot': {
                        title: 'Residuals Plot',
                        purpose: 'Shows prediction errors (residuals = actual - predicted) to identify model biases and heteroscedasticity.',
                        theory: 'Residuals reveal model performance patterns. Random scatter = good model. Patterns (trends, clusters) = model issues. Constant variance = homoscedasticity.',
                        algorithm: 'Residual calculation: e_i = y_i - ŷ_i where y = actual, ŷ = predicted. Plot e_i vs ŷ_i or e_i vs index. Check for patterns, outliers, and variance.',
                        equation: '$$e_i = y_i - \\hat{y}_i$$ where $y_i$ = actual value, $\\hat{y}_i$ = predicted value. Ideal: $e_i \\sim N(0, \\sigma^2)$ (normally distributed, zero mean, constant variance)',
                        interpretation: 'Random scatter around zero = good model. Trends = systematic bias. Funnel shape = heteroscedasticity (varying variance). Outliers = prediction errors.',
                        analysis: 'Check residual distribution (should be normal). Identify patterns indicating model misspecification. Use for model diagnostics and improvement. Low residuals = accurate predictions.'
                    },
                    'default': {
                        title: 'Line Chart',
                        purpose: 'Time series visualization of stellar properties.',
                        theory: 'Line charts reveal temporal patterns and relationships in stellar data.',
                        algorithm: 'Linear interpolation between data points',
                        equation: '$$y(t) = f(x(t))$$',
                        interpretation: 'Trends indicate systematic changes. Variations show variability.',
                        analysis: 'Identify patterns, cycles, and anomalies in temporal data.'
                    }
                },
                'bar-chart': {
                    'feature-importance': {
                        title: 'Feature Importance (ML Models)',
                        purpose: 'Shows which stellar features (temperature, mass, radius, etc.) are most important for age prediction in XGBoost/LightGBM models.',
                        theory: 'Feature importance quantifies each feature\'s contribution to model predictions. Higher importance = stronger predictive power. Helps understand what drives stellar age.',
                        algorithm: 'XGBoost importance: Gain-based (total improvement in loss from splits using feature) or Split-based (number of times feature used). LightGBM: Similar gain-based importance. Normalized to sum to 1.',
                        equation: 'Gain importance: $I_j = \\frac{1}{K} \\sum_{k=1}^{K} \\sum_{s \\in S_j} \\text{gain}_s$ where $K$ = trees, $S_j$ = splits using feature $j$, gain = loss reduction. Normalized: $I_j^{\\text{norm}} = \\frac{I_j}{\\sum_{i=1}^{p} I_i}$',
                        interpretation: 'Taller bars = more important features. Features with high importance strongly influence age predictions. Low importance features may be redundant.',
                        analysis: 'Identify key stellar properties for age estimation. Validate against astrophysical knowledge (e.g., temperature, mass should be important). Use for feature selection and model interpretation.'
                    },
                    'default': {
                        title: 'Bar Chart',
                        purpose: 'Categorical comparison visualization for stellar data.',
                        theory: 'Bar charts provide categorical comparison, making it easy to identify highest/lowest values.',
                        algorithm: 'Bar rendering based on value magnitude',
                        equation: '$$\\text{Bar height} \\propto \\text{value}$$',
                        interpretation: 'Relative bar heights show comparative magnitudes. Tallest bars represent highest values.',
                        analysis: 'Identify top/bottom performers, outliers, and distribution patterns across categories.'
                    }
                },
                'scatter-plot': {
                    'color-rotation': {
                        title: 'Color Index vs Rotation Period',
                        purpose: 'Visualize relationship between stellar color index (B-V or BP-RP) and rotation period using polynomial regression. Cooler (redder) stars typically rotate slower due to magnetic braking and age.',
                        theory: 'Gyrochronology: Stellar rotation slows with age due to magnetic braking. Color index (B-V or BP-RP) correlates with temperature and age. Cooler stars are older and rotate slower. The relationship follows a power-law or polynomial form, revealing stellar evolution stages.',
                        algorithm: 'Polynomial regression (degree 2) fitted on (color_index, rotation_period) pairs. Model: P = a₀ + a₁×C + a₂×C² where P = rotation period, C = color index. Fit using least squares: minimize Σ(P_i - P_predicted)². R² computed for model quality.',
                        equation: '$$P(C) = a_0 + a_1 \\times C + a_2 \\times C^2$$ where $P$ = rotation period (days), $C$ = color index (B-V or BP-RP), $a_0, a_1, a_2$ = polynomial coefficients. R² = $1 - \\frac{\\sum(P_i - \\hat{P}_i)^2}{\\sum(P_i - \\bar{P})^2}$ measures fit quality.',
                        interpretation: 'Positive slope (a₁ > 0) = redder (cooler) stars rotate slower, consistent with age. Quadratic term (a₂) captures non-linearity. High R² (>0.7) = strong relationship. Outliers may be rapid rotators, binary systems, or measurement errors. Clusters indicate stellar populations with similar ages.',
                        analysis: 'Compare fitted curve to theoretical gyrochronology models. Identify outliers (rapid rotators, unusual stars). High R² indicates reliable age-rotation relationship. Use for stellar age estimation via rotation period. Deviations may reveal interesting astrophysics (magnetic fields, binarity).'
                    },
                    'default': {
                        title: 'Scatter Plot',
                        purpose: 'Shows relationship between two stellar properties.',
                        theory: 'Scatter plots reveal correlations, clusters, and outliers in multi-dimensional data.',
                        algorithm: 'Point rendering with optional regression line',
                        equation: '$$y = f(x) + \\varepsilon$$ where $\\varepsilon$ = noise',
                        interpretation: 'Positive slope = positive correlation. Negative = negative correlation. Clusters = groups with similar properties.',
                        analysis: 'Calculate correlation coefficient. Identify outliers. Fit regression models.'
                    }
                },
                'network-graph': {
                    'sky-network': {
                        title: 'Sky Network / Stellar Graph',
                        purpose: 'Network visualization of stellar relationships using 2D embeddings (PCA) from multi-dimensional feature space. Nodes represent stars, edges connect k-nearest neighbors based on feature similarity.',
                        theory: 'PCA reduces high-dimensional stellar features (temperature, mass, radius, luminosity, etc.) to 2D for visualization. Network edges connect similar stars (k-nearest neighbors), revealing stellar associations, clusters, and hierarchical structures. Central nodes may be cluster centers or important stars.',
                        algorithm: 'PCA: X_2d = PCA(X_scaled) where X_scaled = StandardScaler(X_features). Network: For each node, connect to k=3 nearest neighbors. Edge weight = 1/(1 + distance). Force-directed layout positions nodes.',
                        equation: 'PCA: $X_{2D} = X_{scaled} \\times W$ where $W$ = principal components (eigenvectors of covariance matrix). Distance: $d_{ij} = ||x_i - x_j||_2$. Edge weight: $w_{ij} = \\frac{1}{1 + d_{ij}}$ for k-nearest neighbors.',
                        interpretation: 'Dense clusters = stellar associations or similar properties. Central nodes = important stars (cluster centers, binaries, unusual stars). Isolated nodes = field stars or outliers. Edge thickness indicates similarity strength.',
                        analysis: 'Identify stellar clusters, associations, and hierarchical structures. Use for cluster detection and classification. Analyze node centrality to find important stars. Compare network structure to physical stellar associations.'
                    },
                    'default': {
                        title: 'Network Graph',
                        purpose: 'Network visualization of stellar relationships.',
                        theory: 'Networks reveal structure and relationships in stellar data.',
                        algorithm: 'Graph layout algorithms',
                        equation: '$$G = (V, E)$$ where $V$ = vertices (nodes), $E$ = edges (relationships)',
                        interpretation: 'Clusters = groups. Central nodes = important entities.',
                        analysis: 'Identify communities and key entities in the network.'
                    }
                },
                'data-table': {
                    'star-table': {
                        title: 'Star Data Table',
                        purpose: 'Tabular display of stellar properties enabling detailed analysis and filtering from observational datasets (e.g., NASA Exoplanet Archive).',
                        theory: 'Tables provide precise numerical access to stellar parameters for detailed analysis. Data from real astronomical surveys and catalogs.',
                        algorithm: 'Tabular rendering with sortable/filterable columns. Data loaded from CSV/FITS files. Filtering: WHERE conditions on columns (rotation_period, mass, metallicity, cluster, etc.)',
                        equation: 'Various stellar parameters: Mass ($M_\\odot$), Radius ($R_\\odot$), Temperature (K), Luminosity ($L_\\odot$), Age (Gyr), etc. Filtering: $\\text{Filtered} = \\{s \\in S : \\text{condition}(s)\\}$',
                        interpretation: 'Sort by different parameters to identify extremes. Filter to focus on specific stellar types (e.g., main-sequence, giants, clusters).',
                        analysis: 'Compare stellar properties. Identify unusual stars (outliers). Filter for specific research questions (e.g., fast rotators, high-mass stars, cluster members).'
                    },
                    'default': {
                        title: 'Data Table',
                        purpose: 'Structured display of stellar data.',
                        theory: 'Tables enable precise data access and comparison.',
                        algorithm: 'Row/column rendering',
                        equation: 'N/A',
                        interpretation: 'Each row = one star. Each column = one property.',
                        analysis: 'Use sorting and filtering for detailed analysis.'
                    }
                },
                'pie-chart': {
                    'cluster-distribution': {
                        title: 'Cluster Distribution',
                        purpose: 'Shows proportional distribution of stars across identified clusters using K-means, DBSCAN, or HDBSCAN algorithms.',
                        theory: 'Cluster distribution reveals population structure. Dominant clusters indicate major stellar groups. Clustering performed on stellar embeddings or feature space.',
                        algorithm: 'K-means: Minimize within-cluster sum of squares. DBSCAN: Density-based clustering with eps and min_samples. HDBSCAN: Hierarchical density-based clustering. Count stars per cluster label.',
                        equation: 'K-means: $\\text{argmin}_C \\sum_{i=1}^{k} \\sum_{x \\in C_i} ||x - \\mu_i||^2$ where $C_i$ = cluster $i$, $\\mu_i$ = centroid. Cluster%: $\\frac{|C_i|}{N} \\times 100\\%$ where $N$ = total stars',
                        interpretation: 'Large slices = major clusters. Small slices = minor groups. Field stars = unclustered (label = -1 for DBSCAN/HDBSCAN).',
                        analysis: 'Assess cluster significance. Identify dominant populations. Compare to theoretical expectations. Validate cluster quality using silhouette score or within-cluster variance.'
                    },
                    'default': {
                        title: 'Pie Chart',
                        purpose: 'Proportional distribution visualization.',
                        theory: 'Pie charts show part-to-whole relationships.',
                        algorithm: 'Circular sector rendering',
                        equation: '$$\\text{Slice\\%} = \\frac{\\text{Value}}{\\text{Total}} \\times 100\\%$$',
                        interpretation: 'Larger slices = larger proportions.',
                        analysis: 'Identify dominant categories and balance.'
                    }
                }
            }
        ,
            // Common fallbacks to ensure every help flag shows an algorithm/equation
            'common': {
                'streaming-chart': {
                    'default': {
                        title: 'Streaming Chart',
                        purpose: 'Displays live or frequently updating metrics (e.g., rolling volatility or risk scores) pulled from the backend streaming endpoints.',
                        theory: 'Rolling statistics smooth short-term noise while retaining responsiveness to regime shifts.',
                        algorithm: 'Compute rolling window metric: m_t = f(x_{t-w:t}) where f could be std (volatility), mean, or correlation. Update chart each interval with new data from /api/{domain}/stream endpoints.',
                        equation: '$$\\sigma_{t}^{\\text{roll}} = \\sqrt{\\frac{1}{w-1} \\sum_{i=t-w+1}^{t} (r_i - \\bar{r})^2}$$ where $w$ = window length, $r_i$ = returns.',
                        interpretation: 'Rising rolling metric = increasing instability. Falling = calming markets or stable signals.',
                        analysis: 'Use to spot spikes (anomalies) and trend changes. Cross-check against events and news.'
                    }
                },
                'network-graph': {
                    'default': {
                        title: 'Network Graph',
                        purpose: 'Visualizes relationships between entities using weights (correlation or distance) derived in the backend.',
                        theory: 'Edges capture similarity/relationship strength; nodes represent assets or stars.',
                        algorithm: 'Build adjacency where weight_ij = f(similarity_ij). For finance, similarity = |corr(r_i, r_j)|; for astronomy, similarity = 1/(1 + distance). Show edges with |weight| above threshold.',
                        equation: '$$w_{ij} = |\\rho_{ij}| \\quad \\text{or} \\quad w_{ij} = \\frac{1}{1 + d_{ij}}$$ where $\\rho_{ij}$ = correlation, $d_{ij}$ = distance in embedding space.',
                        interpretation: 'Thicker edges = stronger relationships. Clusters = tightly linked groups.',
                        analysis: 'Use to find diversification gaps (finance) or stellar groupings (astronomy).'
                    }
                }
            }
        };
    }

    /**
     * Get explanation for a component
     */
    getExplanation(domain, componentType, componentId) {
        const domainExplanations = this.explanations[domain];
        if (!domainExplanations) {
            return this._getCommonExplanation(componentType, componentId);
        }

        const typeExplanations = domainExplanations[componentType];
        if (!typeExplanations) {
            return this._getCommonExplanation(componentType, componentId);
        }

        // Try specific ID first, then default
        return typeExplanations[componentId] || typeExplanations['default'] || this._getCommonExplanation(componentType, componentId);
    }

    _getCommonExplanation(componentType, componentId) {
        const common = this.explanations['common'];
        if (!common) return null;
        const typeExplanations = common[componentType];
        if (!typeExplanations) return null;
        return typeExplanations[componentId] || typeExplanations['default'] || null;
    }

    /**
     * Show help for a component
     */
    showComponentHelp(domain, componentType, componentId, title) {
        const explanation = this.getExplanation(domain, componentType, componentId);
        if (!explanation) {
            // Fallback to generic explanation
            this.showGenericHelp(componentType, title);
            return;
        }

        this.createComponentModal(explanation, title);
    }

    /**
     * Show generic help
     */
    showGenericHelp(componentType, title) {
        const explanation = {
            title: title || componentType,
            purpose: `This ${componentType} component displays data visualization or metrics.`,
            theory: 'Component theory depends on specific implementation.',
            algorithm: 'Rendering algorithm varies by component type.',
            equation: 'N/A',
            interpretation: 'Interpret values based on context and domain knowledge.',
            analysis: 'Use interactive features to explore the data.'
        };
        this.createComponentModal(explanation, title);
    }

    /**
     * Create component help modal
     */
    createComponentModal(explanation, title) {
        // Remove existing modal
        const existing = document.getElementById('component-help-modal');
        if (existing) {
            existing.remove();
        }

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'component-help-modal';
        modal.className = 'help-modal';
        modal.innerHTML = `
            <div class="help-modal-overlay"></div>
            <div class="help-modal-content" style="max-width: 700px;">
                <div class="help-modal-header">
                    <h3>${explanation.title || title}</h3>
                    <button class="help-modal-close" id="component-help-close">
                        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828842.png" alt="Close" style="width: 20px; height: 20px;">
                    </button>
                </div>
                <div class="help-modal-body">
                    <div class="help-section">
                        <h4>Purpose</h4>
                        <p>${explanation.purpose}</p>
                    </div>
                    <div class="help-section">
                        <h4>Theory</h4>
                        <p>${explanation.theory}</p>
                    </div>
                    <div class="help-section">
                        <h4>Algorithm</h4>
                        <p><code style="background: var(--bg-secondary); padding: 0.25rem 0.5rem; border-radius: 3px; font-family: 'Courier New', monospace;">${explanation.algorithm}</code></p>
                    </div>
                    <div class="help-section">
                        <h4>Equation</h4>
                        <div class="equation-container">${explanation.equation}</div>
                    </div>
                    <div class="help-section">
                        <h4>Interpretation</h4>
                        <p>${explanation.interpretation}</p>
                    </div>
                    <div class="help-section">
                        <h4>Analytical Explanation</h4>
                        <p>${explanation.analysis}</p>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        const closeBtn = document.getElementById('component-help-close');
        const overlay = modal.querySelector('.help-modal-overlay');
        
        const closeModal = () => {
            modal.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => {
                if (modal.parentNode) {
                    modal.remove();
                }
            }, 300);
        };

        if (closeBtn) {
            closeBtn.addEventListener('click', closeModal);
        }
        if (overlay) {
            overlay.addEventListener('click', closeModal);
        }

        // Show modal with animation
        setTimeout(() => {
            modal.style.opacity = '1';
            
            // Render MathJax equations after modal is visible
            if (window.MathJax && window.MathJax.typesetPromise) {
                window.MathJax.typesetPromise([modal]).catch((err) => {
                    console.error('MathJax rendering error:', err);
                });
            }
        }, 10);
        
        // Close on Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }
}

// Global instance
window.componentHelp = new ComponentHelp();
