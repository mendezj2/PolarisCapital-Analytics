# Streaming Analytics Dashboard

## Purpose
Real-time risk and volatility monitoring from Kafka streaming pipeline.

## Layout
- **Row 1**: KPI (Streaming Status) + Line Chart (Real-Time Volatility, spans 2 columns)
- **Row 2**: Gauge (Current Risk) + Data Table (Latest Risk Scores, spans 2 columns)
- **Row 3**: Network Graph (Live Correlation Network, full width)

## Components

### KPI Card
- **Streaming Status**
  - Indicates if streaming pipeline is active
  - Data source: `/api/finance/stream/latest`
  - TODO (USER): Add streaming indicator (green pulsing dot)
  - Display: "Streaming Active" or "Streaming Paused"

### Line Chart
- **Real-Time Volatility**
  - Live volatility updates
  - Data source: `/api/finance/stream/risk` (polling every 2 seconds)
  - TODO (USER): Implement WebSocket for push updates
  - TODO (USER): Show rolling window (last 60 minutes)
  - Multiple series: Per-ticker volatility

### Gauge Card
- **Current Risk**
  - Latest portfolio risk score
  - Data source: `/api/finance/stream/latest`
  - Updates: Every 2-5 seconds
  - TODO (USER): Add alert thresholds

### Data Table
- **Latest Risk Scores**
  - Most recent risk scores per ticker
  - Data source: `/api/finance/stream/risk`
  - Columns: Ticker, Risk Score, Volatility, Anomaly Score, Timestamp
  - TODO (USER): Auto-refresh, highlight changes
  - TODO (USER): Sort by risk score or timestamp

### Network Graph
- **Live Correlation Network**
  - Real-time correlation graph
  - Data source: `/api/finance/stream/graph`
  - Updates: Every 30-60 seconds
  - TODO (USER): Animate edge weight changes
  - TODO (USER): Color nodes by risk level
  - TODO (USER): Highlight anomaly nodes

## Data Flow
1. Kafka producer → risk.processed topic
2. API polls/streams from topic
3. Dashboard components update in real-time
4. TODO (USER): Implement WebSocket for lower latency

## Implementation TODOs
- [ ] Connect to Kafka streaming endpoints
- [ ] Implement WebSocket server for push updates
- [ ] Add streaming data buffering
- [ ] Create anomaly alert system
- [ ] Add streaming pause/resume controls
- [ ] Implement data replay functionality




