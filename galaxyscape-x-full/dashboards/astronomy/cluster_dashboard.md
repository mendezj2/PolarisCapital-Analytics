# Cluster Analysis Dashboard

## Purpose
Visualize and analyze stellar clusters using network graphs and statistics.

## Layout
- **Row 1**: Network Graph (full width, spans 2 columns)
- **Row 2**: Leaderboard (Top Clusters) + Bar Chart (Cluster Analysis)

## Components

### Network Graph
- **Cluster Network**
  - Visual representation of cluster relationships
  - Nodes: Individual stars or cluster centroids
  - Edges: Similarity or proximity relationships
  - TODO (USER): Implement D3.js force layout or ECharts graph
  - TODO (USER): Add node clustering based on cluster labels
  - Interactive: Click nodes to see details, drag to rearrange

### Leaderboard
- **Top Clusters by Size**
  - Ranked list of largest clusters
  - Display: Rank, Cluster ID, Size, Average properties
  - TODO (USER): Add click-to-filter network graph

### Bar Chart
- **Cluster Analysis**
  - Breakdown of cluster properties
  - TODO (USER): Show average age, luminosity, temperature per cluster
  - X-axis: Cluster labels
  - Y-axis: Metric values (age, luminosity, etc.)

## Data Sources
- `/api/astronomy/dashboard/network?type=clusters`
- `/api/astronomy/dashboard/leaderboard?metric=cluster_size`
- `/api/astronomy/dashboard/trends?metric=cluster_analysis`

## Implementation TODOs
- [ ] Build cluster network graph data structure
- [ ] Implement force-directed layout
- [ ] Add cluster property aggregations
- [ ] Create interactive filtering (click leaderboard → highlight in graph)
- [ ] Add cluster comparison tool




