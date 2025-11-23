# Sky Network Dashboard

## Purpose
Visualize stellar relationships as a network graph with connectivity metrics.

## Layout
- **Row 1**: Network Graph (full width, spans 2 columns)
- **Row 2**: Leaderboard (Most Connected Stars) + KPI (Network Metrics)

## Components

### Network Graph
- **Stellar Network Graph**
  - Force-directed graph of star relationships
  - Nodes: Stars (size = importance/centrality)
  - Edges: Similarity relationships (width = strength)
  - TODO (USER): Implement D3.js force simulation
  - TODO (USER): Color nodes by cluster membership
  - TODO (USER): Add zoom, pan, node selection
  - Interactive: Hover for details, click to highlight connections

### Leaderboard
- **Most Connected Stars**
  - Ranked by degree centrality or betweenness
  - Display: Rank, Star ID, Connection Count, Centrality Score
  - TODO (USER): Click to highlight star in network graph

### KPI Card
- **Network Metrics**
  - Key network statistics
  - TODO (USER): Display average degree, clustering coefficient, diameter
  - Metrics: Total nodes, Total edges, Average degree, Clustering coefficient

## Data Sources
- `/api/astronomy/dashboard/network?type=stellar`
- `/api/astronomy/dashboard/leaderboard?metric=connectivity`
- `/api/astronomy/dashboard/kpi?metric=network`

## Implementation TODOs
- [ ] Build stellar similarity graph
- [ ] Calculate network centrality metrics
- [ ] Implement interactive network visualization
- [ ] Add network analysis tools (community detection, path finding)
- [ ] Create network export (GraphML, JSON)




