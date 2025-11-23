# Astronomy Overview Dashboard

## Purpose
High-level overview of astronomy dataset with key metrics and trends.

## Layout
Grid layout with 2 rows:
- **Row 1**: 4 KPI cards (Total Stars, Average Age, Clusters Found, Anomalies)
- **Row 2**: 2 charts side-by-side (Age Distribution, Cluster Sizes)
- **Row 3**: Gauge and Pie chart (Data Quality, Cluster Distribution)

## Components

### KPI Cards
1. **Total Stars**
   - Data source: `/api/astronomy/dashboard/kpi?metric=total_stars`
   - TODO (USER): Implement aggregation query
   - Display: Count with trend indicator

2. **Average Age**
   - Data source: `/api/astronomy/dashboard/kpi?metric=avg_age`
   - TODO (USER): Calculate mean stellar age
   - Display: Value in millions of years

3. **Clusters Found**
   - Data source: `/api/astronomy/dashboard/kpi?metric=clusters`
   - TODO (USER): Count unique cluster labels
   - Display: Number with change indicator

4. **Anomalies**
   - Data source: `/api/astronomy/dashboard/kpi?metric=anomalies`
   - TODO (USER): Count anomaly flags
   - Display: Count with alert styling

### Charts
1. **Age Distribution Over Time** (Line Chart)
   - Component: `line-chart.js` with ECharts
   - Data source: `/api/astronomy/dashboard/trends?metric=age`
   - TODO (USER): Implement time-series aggregation
   - X-axis: Time periods
   - Y-axis: Age values

2. **Cluster Sizes** (Bar Chart)
   - Component: `bar_chart.js` with ECharts
   - Data source: `/api/astronomy/dashboard/trends?metric=clusters`
   - TODO (USER): Group by cluster label, count stars
   - X-axis: Cluster labels
   - Y-axis: Star count

3. **Cluster Distribution** (Pie Chart)
   - Component: `pie_chart.js` with ECharts
   - Data source: `/api/astronomy/dashboard/trends?metric=cluster_dist`
   - TODO (USER): Calculate percentage distribution
   - Show: Cluster name and percentage

### Gauge
- **Data Quality Score**
  - Component: `gauge_card.js` with ECharts
  - Data source: `/api/astronomy/dashboard/kpi?metric=data_quality`
  - TODO (USER): Calculate quality metrics (completeness, validity)
  - Range: 0-100
  - Colors: Red (0-50), Yellow (50-80), Green (80-100)

## Data Flow
1. Dashboard loads → Fetch all KPI/metric data
2. User uploads new CSV → Refresh all components
3. TODO (USER): Add auto-refresh interval for live updates

## Implementation TODOs
- [ ] Implement KPI aggregation endpoints
- [ ] Create time-series data processing
- [ ] Add ECharts configuration for each chart type
- [ ] Implement data quality scoring algorithm
- [ ] Add loading states and error handling
- [ ] Add export functionality (PDF, PNG)




