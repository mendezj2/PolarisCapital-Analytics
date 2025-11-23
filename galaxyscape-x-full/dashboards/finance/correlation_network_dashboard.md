# Correlation Network Dashboard

## Purpose
Visualize asset correlations as a network graph with sector clustering.

## Layout
- **Row 1**: Network Graph (full width, spans 2 columns)
- **Row 2**: Leaderboard (Most Correlated Pairs) + Bar Chart (Correlation Heatmap)

## Components

### Network Graph
- **Asset Correlation Network**
  - Force-directed graph of asset correlations
  - Nodes: Assets/tickers (size = market cap or risk)
  - Edges: Correlation relationships (width = |correlation|, color = sign)
  - TODO (USER): Implement ECharts graph or D3.js force layout
  - TODO (USER): Cluster nodes by sector
  - TODO (USER): Color nodes by risk level
  - Interactive: Click nodes to see details, filter by correlation threshold

### Leaderboard
- **Most Correlated Pairs**
  - Ranked list of highest correlation pairs
  - Data source: `/api/finance/dashboard/leaderboard?metric=correlation`
  - Display: Rank, Pair (Ticker1-Ticker2), Correlation, Sector
  - TODO (USER): Click to highlight pair in network graph

### Bar Chart / Heatmap
- **Correlation Matrix Visualization**
  - Alternative view of correlations
  - Data source: `/api/finance/dashboard/trends?metric=correlation_matrix`
  - TODO (USER): Use ECharts heatmap or custom visualization
  - X-axis: Tickers
  - Y-axis: Tickers
  - Color: Correlation value (-1 to +1)

## Data Sources
- `/api/finance/dashboard/network?type=correlation`
- `/api/finance/dashboard/leaderboard?metric=correlation`
- `/api/finance/dashboard/trends?metric=correlation_matrix`

## Implementation TODOs
- [ ] Calculate rolling correlations
- [ ] Build correlation graph data structure
- [ ] Implement sector-based clustering
- [ ] Add correlation threshold filtering
- [ ] Create correlation export (CSV, JSON)
- [ ] Add correlation analysis tools (find clusters, detect breaks)




