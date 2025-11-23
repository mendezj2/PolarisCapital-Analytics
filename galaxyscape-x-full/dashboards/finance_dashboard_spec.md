# Finance Dashboard Specification

## Layout
1. Header KPIs: VaR/CVaR, max drawdown, Sharpe ratio
2. Left column: Risk-return scatter, volatility chart
3. Right column: Portfolio allocation, SHAP importance
4. Bottom: Anomaly heatmap + market network

## Data Sources
- Snowflake views: VW_FINANCE_RISK_SUMMARY, MARKET_FACT
- API endpoints: /api/finance/graph, /api/finance/risk_report

## Components
1. Risk score gauge (0-100)
2. Volatility time series chart
3. Risk-return scatter plot
4. Portfolio breakdown treemap
5. SHAP feature importance bars
6. Anomaly heatmap




