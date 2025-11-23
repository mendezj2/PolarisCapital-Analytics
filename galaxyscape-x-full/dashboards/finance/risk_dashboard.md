# Risk Dashboard

## Purpose
Comprehensive risk analysis with portfolio-level and asset-level risk metrics.

## Layout
Grid layout with 4 rows:
- **Row 1**: 3 components (Portfolio Risk KPI, VaR KPI, Risk Level Gauge)
- **Row 2**: Line Chart (Risk Trends, spans 2 columns)
- **Row 3**: Bar Chart (Risk by Asset, spans 2 columns)
- **Row 4**: Leaderboard (Top Risky Assets) + Data Table (Risk Details)

## Components

### KPI Cards
1. **Portfolio Risk Score**
   - Aggregated risk across entire portfolio
   - Data source: `/api/finance/dashboard/kpi?metric=portfolio_risk`
   - TODO (USER): Calculate weighted average risk
   - Display: Score 0-100 with color coding

2. **VaR (95%)**
   - Value at Risk at 95% confidence
   - Data source: `/api/finance/dashboard/kpi?metric=var`
   - TODO (USER): Implement VaR calculation
   - Display: Dollar amount or percentage

### Gauge Card
- **Risk Level**
  - Overall portfolio risk level
  - Data source: `/api/finance/dashboard/kpi?metric=risk_level`
  - TODO (USER): Map risk score to level (Low/Medium/High)
  - Colors: Green (Low), Yellow (Medium), Red (High)

### Line Chart
- **Risk Trends Over Time**
  - Historical risk progression
  - Data source: `/api/finance/dashboard/trends?metric=risk`
  - TODO (USER): Aggregate risk by time period
  - Multiple series: Portfolio risk, Market risk, Idiosyncratic risk

### Bar Chart
- **Risk by Asset**
  - Individual asset risk scores
  - Data source: `/api/finance/dashboard/trends?metric=risk_breakdown`
  - TODO (USER): Sort by risk score (highest first)
  - X-axis: Asset tickers
  - Y-axis: Risk score

### Leaderboard
- **Top Risky Assets**
  - Ranked list of highest risk assets
  - Data source: `/api/finance/dashboard/leaderboard?metric=risk`
  - Display: Rank, Ticker, Risk Score, Change

### Data Table
- **Risk Details**
  - Comprehensive risk breakdown
  - Data source: `/api/finance/dashboard/trends?metric=risk_table`
  - Columns: Ticker, Risk Score, Volatility, Beta, VaR, CVaR
  - TODO (USER): Add sorting, filtering, export

## Data Sources
All endpoints under `/api/finance/dashboard/*`

## Implementation TODOs
- [ ] Implement VaR and CVaR calculations
- [ ] Create risk aggregation logic
- [ ] Add risk decomposition (systematic vs idiosyncratic)
- [ ] Implement risk stress testing
- [ ] Add risk reporting export




