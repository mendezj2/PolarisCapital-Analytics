# Finance Streaming Dashboard Specification

## Layout
- Header: Portfolio name, last update timestamp
- Risk Gauge: Circular gauge (0-100)
- Volatility Chart: Time series
- Correlation Network: Live graph
- Recent Anomalies Table

## Components
1. Risk Score Gauge - displays portfolio-wide risk
2. Real-Time Volatility Chart - last 60 minutes
3. Correlation Network Graph - updated every 30-60 seconds
4. Recent Anomalies Table - sorted by anomaly score

## Data Sources
- /api/finance/stream/risk
- /api/finance/stream/graph
- /api/finance/stream/latest

## Update Frequency
- Risk gauge: Every 2 seconds
- Volatility chart: Every 1-2 seconds
- Network graph: Every 30-60 seconds
- Anomalies table: Every 5 seconds




