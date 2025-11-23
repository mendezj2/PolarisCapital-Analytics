# Full User Interactivity - Filter Implementation

## Overview
This document describes the comprehensive filter system implemented for both Astronomy and Finance dashboards, providing full user interactivity with real-time updates.

## Filter Types Implemented

### 1. Date Range Filters
- **Type**: `date-range`
- **Usage**: Filter time-series data by date range
- **Example**: Finance risk trends, portfolio returns, astronomy light curves
- **Implementation**: Two date inputs (start/end) that update `date_start` and `date_end` in filter payload

### 2. Numeric Sliders
- **Type**: `slider`
- **Usage**: Adjust thresholds and ranges (risk scores, correlation, volatility, temperature, etc.)
- **Features**: Min/max values, step size, real-time value display
- **Example**: Correlation threshold (0-1), risk score (0-100), temperature ranges

### 3. Multi-Select Dropdowns
- **Type**: `multi-select`
- **Usage**: Select multiple items from a list (sectors, clusters, tickers)
- **Features**: Multiple selection, visual feedback
- **Example**: Finance sectors (Technology, Finance, Retail, Energy), Astronomy clusters

### 4. Single-Select Dropdowns
- **Type**: `dropdown`
- **Usage**: Select one option from a list
- **Example**: Status (compliant/non-compliant), risk level (Low/Medium/High/Critical)

### 5. Checkboxes
- **Type**: `checkbox`
- **Usage**: Multiple independent selections
- **Example**: Sector selection in finance, cluster selection in astronomy

### 6. Toggle Switches
- **Type**: `toggle`
- **Usage**: Binary on/off options
- **Example**: "Use ML Predictions", "Show Anomalies Only", "Log/Linear Scale"
- **Features**: Visual toggle with on/off labels

### 7. Search Inputs
- **Type**: `search`
- **Usage**: Text search with debouncing
- **Example**: Search ticker symbols, star names/IDs
- **Features**: Debounced input (300ms default), search icon

### 8. Number Range Filters
- **Type**: `number`
- **Usage**: Min/max numeric ranges
- **Example**: Rotation period, mass, metallicity, price ranges
- **Features**: Two inputs (min/max) for range selection

## Backend Filter Endpoints

### Finance Endpoints

#### `/api/finance/filter` (POST)
- **Purpose**: Apply filters and return summary
- **Input**: JSON with filter parameters
- **Output**: Filtered data summary (count, tickers, date range, risk scores, anomalies)
- **Filters Supported**:
  - `date_start`, `date_end`: Date range
  - `tickers`: String (search) or array (multi-select)
  - `sectors`: Array of sector names
  - `risk_min`, `risk_max`: Risk score range
  - `volatility_min`, `volatility_max`: Volatility range
  - `correlation_threshold`: Correlation threshold
  - `price_min`, `price_max`: Price range

#### `/api/finance/get_filtered_data` (POST)
- **Purpose**: Get filtered data for specific component types
- **Input**: JSON with filters + `data_type` (chart, table, kpi, network)
- **Output**: Formatted data for the requested component type
- **Data Types**:
  - `chart`: Time series data (xAxis, series)
  - `table`: Table data (columns, data rows)
  - `kpi`: KPI metrics (risk_score, var_95, num_assets, avg_volatility)
  - `network`: Network graph data (nodes, edges)

### Astronomy Endpoints

#### `/api/astronomy/filter` (POST)
- **Purpose**: Apply filters and return summary
- **Input**: JSON with filter parameters
- **Output**: Filtered data summary (count, stars, clusters)
- **Filters Supported**:
  - `rotation_min`, `rotation_max`: Rotation period range
  - `color_min`, `color_max`: Color index range
  - `mass_min`, `mass_max`: Mass range
  - `metallicity_min`, `metallicity_max`: Metallicity range
  - `temperature_min`, `temperature_max`: Temperature range
  - `age_min`, `age_max`: Age range (Gyr)
  - `search`: Star name/ID search
  - `clusters`: Array of cluster names
  - `anomalies_only`: Boolean toggle

#### `/api/astronomy/get_filtered_data` (POST)
- **Purpose**: Get filtered data for specific component types
- **Input**: JSON with filters + `data_type` (table, scatter, sky_map, light_curve)
- **Output**: Formatted data for the requested component type
- **Data Types**:
  - `table`: Star catalog data (columns, data rows)
  - `scatter`: Scatter plot data (xAxis, yAxis, data points)
  - `sky_map`: Sky coordinates (RA/Dec, magnitude, temperature)
  - `light_curve`: Time series light curve (times, flux, period)

## Frontend Implementation

### Global Filter Panel
- **Location**: Top of each dashboard (above components)
- **Rendering**: Automatically rendered from `globalFilters` in dashboard config
- **Styling**: Grid layout, responsive, theme-aware
- **Behavior**: All filters update simultaneously, triggering component refreshes

### Component Updates
- **Method**: `updateComponentWithFilters(componentId, compConfig)`
- **Process**:
  1. Determine data type from component type
  2. Build filter payload (flatten nested objects)
  3. POST to `/api/{domain}/get_filtered_data`
  4. Update component using `component.update()` or direct ECharts `setOption()`
- **ECharts Updates**: Use `setOption(option, true)` for full re-render (notMerge=true)

### Filter Change Handling
- **Global Filters**: Update all components with `useFilteredData: true`
- **Component Filters**: Update individual components
- **Debouncing**: Search inputs debounced (300ms default)
- **Real-time**: Sliders and toggles update immediately

## Dashboard Configurations

### Finance Dashboards with Filters

1. **Risk Dashboard**
   - Date range, ticker search, sector multi-select
   - Risk score sliders (min/max)
   - Volatility range
   - ML predictions toggle

2. **Correlation Network**
   - Correlation threshold slider
   - Ticker search
   - Sector multi-select
   - Risk score range

3. **Streaming Dashboard**
   - Ticker search
   - Sector checkboxes
   - Correlation threshold
   - Anomalies-only toggle

4. **Portfolio Dashboard**
   - Date range
   - Ticker search
   - Sector multi-select
   - Expected return slider
   - Time horizon slider

5. **Compliance Dashboard**
   - Status dropdown
   - Risk level dropdown
   - Date range

### Astronomy Dashboards with Filters

1. **Star Explorer**
   - Star name/ID search
   - Rotation period range
   - Color index range
   - Mass range
   - Metallicity range
   - Temperature range
   - Age range
   - Cluster multi-select
   - Anomalies-only toggle

2. **Sky Map**
   - Star search
   - Cluster multi-select
   - Temperature sliders (min/max)
   - Magnitude sliders (min/max)

3. **Light Curve**
   - Star name search
   - Star ID range
   - Period range sliders

4. **Cluster Dashboard**
   - Cluster multi-select
   - Cluster size slider
   - Star search

5. **Anomaly Dashboard**
   - Anomalies-only toggle
   - Anomaly score slider
   - Star search
   - Mass range
   - Temperature range

## Data Flow

1. **User Interaction**: User changes filter value (slider, dropdown, etc.)
2. **Filter Update**: `FilterPanel` updates `currentFilters` object
3. **Callback Trigger**: `onFilterChange` callback fires
4. **Component Update**: `layout_manager.handleGlobalFilterChange()` called
5. **Backend Request**: POST to `/api/{domain}/get_filtered_data` with filters
6. **Data Processing**: Backend applies filters to DataFrame, calculates ML outputs
7. **Response**: JSON with filtered, formatted data
8. **Component Refresh**: Component updates using `update()` or ECharts `setOption()`
9. **Visual Update**: Chart/table/KPI updates instantly with smooth animation

## ML Integration

### Finance ML Outputs
- **Risk Scores**: Calculated from volatility (XGBoost/LightGBM style)
- **Anomaly Detection**: Isolation Forest style (simplified with statistical outliers)
- **Volatility Forecasts**: LSTM-style predictions (simplified with rolling windows)
- **Correlations**: Real correlation matrix from price returns

### Astronomy ML Outputs
- **Stellar Ages**: ML-predicted ages (from age column or calculated)
- **Anomaly Scores**: Statistical outlier detection
- **Embeddings**: UMAP/PCA/t-SNE coordinates (simplified with position data)
- **Cluster Membership**: Cluster assignments

## Technical Details

### ECharts Dynamic Updates
- Use `chart.setOption(option, true)` where `true` = `notMerge`
- This ensures full re-render with new data
- Smooth animations enabled by default
- Resize handlers maintain responsiveness

### Filter Payload Structure
```javascript
{
  data_type: 'chart' | 'table' | 'kpi' | 'network',
  date_start: '2023-01-01',
  date_end: '2024-01-01',
  tickers: 'AAPL' | ['AAPL', 'MSFT'],
  sectors: ['Technology', 'Finance'],
  risk_min: 0,
  risk_max: 100,
  correlation_threshold: 0.5,
  // ... other filters
}
```

### Component Update Methods
- **Charts**: `component.update(data)` or `chart.setOption(option, true)`
- **Tables**: `component.update(data)` or `component.render()`
- **KPIs**: `component.update({ value, change })`
- **Gauges**: `component.update({ value })` or `chart.setOption()`
- **Networks**: `component.update({ nodes, edges })` or `renderECharts()`
- **Leaderboards**: `component.update({ data })` or `component.render()`

## Usage Examples

### Adding Filters to a Dashboard
```javascript
{
  layout: 'grid',
  globalFilters: [
    {
      name: 'risk_threshold',
      label: 'Risk Threshold',
      type: 'slider',
      min: 0,
      max: 100,
      step: 1,
      defaultValue: 50
    },
    {
      name: 'sectors',
      label: 'Sectors',
      type: 'multi-select',
      options: [
        { value: 'Tech', label: 'Technology' },
        { value: 'Finance', label: 'Finance' }
      ],
      defaultValue: []
    }
  ],
  components: [
    {
      type: 'line-chart',
      id: 'chart-1',
      title: 'Risk Trends',
      apiEndpoint: '/api/finance/risk_timeseries',
      useFilteredData: true,  // Enable filtered data
      position: { row: 1, col: 1 }
    }
  ]
}
```

## Performance Considerations

- **Debouncing**: Search inputs debounced to reduce API calls
- **Batch Updates**: All components update together when global filters change
- **ECharts Optimization**: Use `notMerge: true` only when necessary
- **Caching**: Consider adding client-side caching for frequently used filters

## Testing

To test the filter system:
1. Navigate to any dashboard
2. Adjust filters in the global filter panel
3. Observe charts/tables/KPIs update in real-time
4. Verify filters persist when switching between dashboards (within same domain)
5. Test all filter types: sliders, dropdowns, toggles, search, date ranges

## Future Enhancements

- Filter presets/saved configurations
- URL parameters for filter state
- Filter validation and error handling
- Advanced filters (regex, custom queries)
- Filter export/import
- Collaborative filtering (if multi-user support added)




