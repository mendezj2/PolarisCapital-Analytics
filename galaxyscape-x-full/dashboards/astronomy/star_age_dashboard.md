# Star Age Analysis Dashboard

## Purpose
Detailed analysis of stellar age predictions and distributions.

## Layout
Grid layout:
- **Row 1**: Gauge (Data Quality) + Line Chart (Age Trends, spans 2 columns)
- **Row 2**: Data Table (Age Predictions, spans 2 columns)

## Components

### Gauge Card
- **Data Quality Score**
  - Measures completeness and accuracy of age data
  - TODO (USER): Implement quality scoring algorithm

### Line Chart
- **Age Prediction Trends**
  - Shows how age predictions change over time or across different models
  - TODO (USER): Aggregate predictions by time/model
  - Multiple series: XGBoost, LightGBM, Actual (if available)

### Data Table
- **Age Predictions**
  - Detailed table of individual star age predictions
  - Columns: Star ID, Predicted Age, Actual Age (if available), Error, Model
  - TODO (USER): Add sorting, filtering, pagination
  - TODO (USER): Highlight high-error predictions

## Data Sources
- `/api/astronomy/dashboard/kpi?metric=data_quality`
- `/api/astronomy/dashboard/trends?metric=age_predictions`
- `/api/astronomy/dashboard/trends?metric=age_table`

## Implementation TODOs
- [ ] Implement age prediction aggregation
- [ ] Create comparison view (XGBoost vs LightGBM)
- [ ] Add error analysis (MAE, RMSE per star)
- [ ] Implement table sorting and filtering
- [ ] Add export to CSV functionality




