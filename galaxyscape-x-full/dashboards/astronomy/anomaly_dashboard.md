# Anomaly Detection Dashboard

## Purpose
Identify and analyze anomalous stars in the dataset.

## Layout
- **Row 1**: KPI (Anomalies Detected) + Data Table (Anomaly Details, spans 2 columns)
- **Row 2**: Line Chart (Anomaly Trends, spans 2 columns)

## Components

### KPI Card
- **Anomalies Detected**
  - Total count of flagged anomalies
  - TODO (USER): Aggregate from IsolationForest/LOF results
  - Display: Count with alert styling if high

### Data Table
- **Anomaly Details**
  - Table of all detected anomalies
  - Columns: Star ID, Anomaly Score, Detection Method, Features, Timestamp
  - TODO (USER): Sort by anomaly score (highest first)
  - TODO (USER): Add filter by detection method
  - TODO (USER): Highlight extreme anomalies

### Line Chart
- **Anomaly Detection Over Time**
  - Shows anomaly detection trends
  - TODO (USER): Group by time periods or detection runs
  - Multiple series: IsolationForest count, LOF count, Combined
  - Y-axis: Anomaly count
  - X-axis: Time or detection run number

## Data Sources
- `/api/astronomy/dashboard/kpi?metric=anomalies`
- `/api/astronomy/dashboard/trends?metric=anomaly_table`
- `/api/astronomy/dashboard/trends?metric=anomaly_trends`

## Implementation TODOs
- [ ] Aggregate anomaly detection results
- [ ] Create anomaly scoring visualization
- [ ] Add anomaly investigation workflow
- [ ] Implement anomaly export (for further analysis)
- [ ] Add anomaly validation interface




