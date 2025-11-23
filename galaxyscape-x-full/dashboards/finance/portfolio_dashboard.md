# Portfolio Analysis Dashboard

## Purpose
Comprehensive portfolio performance and allocation analysis.

## Layout
Grid layout:
- **Row 1**: 3 components (Portfolio Value KPI, Sharpe Ratio KPI, Allocation Pie Chart)
- **Row 2**: Line Chart (Portfolio Returns, spans 2 columns)
- **Row 3**: Map Card (Geographic Risk) + Data Table (Holdings)

## Components

### KPI Cards
1. **Portfolio Value**
   - Current total portfolio value
   - Data source: `/api/finance/dashboard/kpi?metric=portfolio_value`
   - TODO (USER): Calculate from holdings and prices
   - Display: Dollar amount with change indicator

2. **Sharpe Ratio**
   - Risk-adjusted return metric
   - Data source: `/api/finance/dashboard/kpi?metric=sharpe`
   - TODO (USER): Calculate Sharpe ratio
   - Display: Ratio value with interpretation

### Pie Chart
- **Portfolio Allocation**
  - Asset allocation breakdown
  - Data source: `/api/finance/dashboard/trends?metric=allocation`
  - TODO (USER): Group by asset or sector
  - Show: Percentage and dollar amount per slice

### Line Chart
- **Portfolio Returns**
  - Historical portfolio performance
  - Data source: `/api/finance/dashboard/trends?metric=returns`
  - TODO (USER): Calculate cumulative returns
  - Multiple series: Total returns, benchmark comparison

### Map Card
- **Geographic Risk Map**
  - Risk visualization by geography
  - Data source: `/api/finance/dashboard/map`
  - TODO (USER): Load ECharts geo map data
  - TODO (USER): Map assets to countries/regions
  - Color: Risk level by region

### Data Table
- **Holdings**
  - Detailed portfolio holdings
  - Data source: `/api/finance/dashboard/trends?metric=holdings`
  - Columns: Ticker, Quantity, Price, Value, Weight, Sector
  - TODO (USER): Add sorting, filtering, export

## Data Sources
All endpoints under `/api/finance/dashboard/*`

## Implementation TODOs
- [ ] Implement portfolio value calculation
- [ ] Calculate Sharpe ratio and other metrics
- [ ] Create allocation aggregation
- [ ] Load geographic mapping data
- [ ] Implement portfolio comparison tools
- [ ] Add portfolio rebalancing suggestions




