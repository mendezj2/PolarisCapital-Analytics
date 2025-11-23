# Astronomy Dashboard Specification

## Layout
- Top row: Upload summary cards
- Middle: Live Sky View + SHAP importance
- Bottom: Cosmic Twin table + cluster distribution

## Data Sources
- Snowflake views: VW_ASTRO_CLUSTER_SUMMARY, STARS_FACT
- API endpoints: /api/astronomy/graph, /api/astronomy/cosmic_twin

## Components
1. Upload summary (row count, columns, domain confidence)
2. Live Sky View visualization
3. SHAP feature importance bar chart
4. Cosmic Twin results table
5. Cluster distribution donut chart




