/**
 * Network Graph Component
 * Displays network/graph visualizations using D3.js or ECharts
 * TODO (USER): Implement network graph visualization
 */

class NetworkGraph {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Network Graph',
            nodes: config.nodes || [],
            edges: config.edges || [],
            apiEndpoint: config.apiEndpoint || null,
            useECharts: config.useECharts || false, // Use ECharts graph instead of D3
            ...config
        };
        this.chart = null;
        this.simulation = null;
    }

    /**
     * Initialize the network graph
     */
    async init() {
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            container = document.querySelector(`#${this.containerId}-container`);
        }
        if (!container) {
            console.warn(`Network graph container ${this.containerId} not found`);
            return;
        }
        
        // Store container reference
        this.container = container;

        // Fetch data from API
        if (this.config.apiEndpoint) {
            try {
                const response = await fetch(this.config.apiEndpoint);
                const data = await response.json();
                
                if (data.nodes && data.edges) {
                    this.config.nodes = data.nodes;
                    this.config.edges = data.edges;
                } else if (data.error) {
                    console.error('API error:', data.error);
                }
            } catch (error) {
                console.error('Failed to fetch network data:', error);
                // Still render with empty/default data
            }
        }

        // Ensure we have at least empty arrays for rendering
        if (!this.config.nodes) this.config.nodes = [];
        if (!this.config.edges) this.config.edges = [];

        // Use ECharts by default for better integration
        this.config.useECharts = true;
        this.renderECharts();
        
        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Render using ECharts graph
     * Fully implemented ECharts graph visualization
     */
    renderECharts() {
        const container = this.container || document.getElementById(this.containerId) || document.querySelector(`#${this.containerId}-container`);
        if (!container) return;

        if (!this.chart) {
            this.chart = echarts.init(container);
        }

        // Convert nodes and edges to ECharts format
        const nodes = this.config.nodes.map(node => ({
            id: node.id,
            name: node.name || node.id,
            value: node.risk_score || node.size || 10,
            category: node.sector || 0,
            symbolSize: node.size || 20,
            itemStyle: {
                color: this._getNodeColor(node)
            }
        }));

        const links = this.config.edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            value: edge.weight || edge.correlation || 0.5,
            lineStyle: {
                width: (edge.weight || 0.5) * 5,
                opacity: 0.6
            }
        }));

        const categories = [...new Set(nodes.map(n => n.category))].map((cat, idx) => ({
            name: cat || 'Default'
        }));

        const option = {
            tooltip: {
                formatter: (params) => {
                    if (params.dataType === 'node') {
                        const node = this.config.nodes.find(n => n.id === params.data.id);
                        return `${params.data.name}<br/>Risk: ${node?.risk_score || 'N/A'}<br/>Sector: ${node?.sector || 'N/A'}`;
                    } else {
                        return `${params.data.source} → ${params.data.target}<br/>Correlation: ${params.data.value.toFixed(3)}`;
                    }
                },
                backgroundColor: 'var(--bg-card)',
                borderColor: 'var(--border-color)',
                textStyle: {
                    color: 'var(--text-primary)',
                    fontSize: 12
                }
            },
            legend: {
                data: categories.map(c => c.name),
                orient: 'vertical',
                left: 'left',
                top: 'middle'
            },
            series: [{
                type: 'graph',
                layout: 'force',
                data: nodes,
                links: links,
                categories: categories,
                roam: true,
                label: {
                    show: true,
                    position: 'right',
                    formatter: '{b}',
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    fontWeight: 500
                },
                lineStyle: {
                    color: 'source',
                    curveness: 0.3
                },
                emphasis: {
                    focus: 'adjacency',
                    lineStyle: {
                        width: 10
                    }
                },
                force: {
                    repulsion: 200,
                    gravity: 0.1,
                    edgeLength: 50,
                    layoutAnimation: true
                }
            }]
        };

        // Ensure we have valid arrays
        if (!this.config.nodes) this.config.nodes = [];
        if (!this.config.edges) this.config.edges = [];
        
        // Ensure chart renders even with empty data
        if (this.config.nodes.length === 0) {
            // Show placeholder message
            option.graphic = [{
                type: 'text',
                left: 'center',
                top: 'middle',
                style: {
                    text: 'No network data available',
                    fontSize: 14,
                    fill: 'var(--text-secondary)'
                }
            }];
        }
        
        this.chart.setOption(option);
        
        // Resize chart to ensure proper rendering
        setTimeout(() => {
            if (this.chart) {
                this.chart.resize();
            }
        }, 100);
    }

    /**
     * Get node color based on risk score or sector
     */
    _getNodeColor(node) {
        if (node.risk_score !== undefined) {
            // Color by risk: green (low) to red (high)
            const risk = node.risk_score;
            if (risk < 33) return '#22c55e';  // Green
            if (risk < 66) return '#fbbf24';  // Yellow
            return '#ef4444';  // Red
        }
        // Color by sector
        const sectorColors = {
            'Technology': '#3b82f6',
            'Finance': '#8b5cf6',
            'Retail': '#ec4899',
            'Energy': '#f59e0b'
        };
        return sectorColors[node.sector] || '#94a3b8';
    }

    /**
     * Render using D3.js force layout
     * TODO (USER): Implement D3.js force-directed graph
     */
    renderD3() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        // TODO (USER): Implement D3.js force simulation
        // Example structure:
        // const width = container.offsetWidth;
        // const height = container.offsetHeight;
        // const svg = d3.select(container).append('svg')
        //     .attr('width', width)
        //     .attr('height', height);
        // 
        // this.simulation = d3.forceSimulation(this.config.nodes)
        //     .force('link', d3.forceLink(this.config.edges).id(d => d.id))
        //     .force('charge', d3.forceManyBody().strength(-300))
        //     .force('center', d3.forceCenter(width / 2, height / 2));
        // 
        // // Add links and nodes
        // // TODO: Complete D3 implementation
    }

    /**
     * Update graph with new data
     */
    update(data) {
        if (data.nodes) this.config.nodes = data.nodes;
        if (data.edges) this.config.edges = data.edges;
        if (this.chart) {
            this.renderECharts();
        } else {
            // Re-initialize if chart doesn't exist
            this.init();
        }
    }

    /**
     * Destroy the chart
     */
    destroy() {
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }

    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = NetworkGraph;
}

