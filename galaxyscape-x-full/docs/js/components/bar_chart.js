/**
 * Bar Chart Component
 * Displays bar charts using ECharts
 * TODO (USER): Implement bar chart visualization with ECharts
 */

class BarChart {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Bar Chart',
            xAxis: config.xAxis || [],
            yAxis: config.yAxis || [],
            data: config.data || [],
            apiEndpoint: config.apiEndpoint || null,
            ...config
        };
        this.chart = null;
    }

    /**
     * Initialize the bar chart
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
            console.warn(`Bar chart container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API if endpoint provided
        if (this.config.apiEndpoint) {
            try {
                // Check if this is a POST endpoint (stock/compare)
                let response;
                if (this.config.apiEndpoint.includes('/stock/compare')) {
                    // Get symbols from stock-symbol-input component
                    const symbolInput = document.querySelector('[data-id="stock-symbol-input"]');
                    let symbols = [];
                    if (symbolInput && typeof StockSymbolInput !== 'undefined') {
                        const component = window.layoutManager?.components?.get('stock-symbol-input');
                        if (component && typeof component.getSymbols === 'function') {
                            symbols = component.getSymbols();
                        }
                    }
                    
                    // If no symbols, use default
                    if (symbols.length === 0) {
                        symbols = ['AAPL', 'MSFT', 'GOOGL'];
                    }
                    
                    response = await fetch(this.config.apiEndpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({tickers: symbols})
                    });
                } else {
                    response = await fetch(this.config.apiEndpoint);
                }
                const result = await response.json();
                
                // Handle different response formats
                if (result.comparison && Array.isArray(result.comparison)) {
                    // Stock comparison format
                    this.config.data = result.comparison.map(item => item.current_price);
                    this.config.xAxis = result.comparison.map(item => item.ticker);
                    this.config.series = [{
                        name: 'Current Price',
                        data: result.comparison.map(item => item.current_price),
                        type: 'bar'
                    }];
                } else if (result.scenarios && Array.isArray(result.scenarios)) {
                    // Future outcomes scenarios
                    this.config.data = result.scenarios.map(s => s.value);
                    this.config.xAxis = result.scenarios.map(s => s.name);
                    this.config.series = [{
                        name: 'Projected Value',
                        data: result.scenarios.map(s => s.value),
                        type: 'bar'
                    }];
                } else if (result.xAxis && result.data) {
                    this.config.xAxis = result.xAxis;
                    this.config.data = result.data;
                } else if (result.series && result.series[0]) {
                    this.config.xAxis = result.xAxis || [];
                    this.config.data = result.series[0].data || [];
                } else if (result.data && Array.isArray(result.data)) {
                    // Scatter plot format (astronomy)
                    this.config.scatterData = result.data;
                    this.config.isScatter = true;
                } else if (result.xgboost && Array.isArray(result.xgboost)) {
                    // Feature importance format (ML models)
                    this.config.xAxis = result.xgboost.map(item => item.feature);
                    this.config.data = result.xgboost.map(item => item.importance);
                } else if (result.lightgbm && Array.isArray(result.lightgbm)) {
                    // Feature importance format (LightGBM)
                    this.config.xAxis = result.lightgbm.map(item => item.feature);
                    this.config.data = result.lightgbm.map(item => item.importance);
                } else if (result.in_store_assets && Array.isArray(result.in_store_assets)) {
                    // Signage evaluation format
                    this.config.xAxis = result.in_store_assets.map(item => item.location);
                    this.config.data = result.in_store_assets.map(item => item.roi);
                } else if (result.predicted_initiatives && Array.isArray(result.predicted_initiatives)) {
                    // Predictability model format
                    this.config.xAxis = result.predicted_initiatives.map(item => item.initiative);
                    this.config.data = result.predicted_initiatives.map(item => item.predicted_roi);
                } else if (result.current_allocation && Array.isArray(result.current_allocation)) {
                    // Expenditure optimization format
                    this.config.xAxis = result.current_allocation.map(item => item.channel);
                    this.config.data = result.current_allocation.map(item => item.roi);
                } else if (result.nash_weights && typeof result.nash_weights === 'object') {
                    // Game theory - Nash equilibrium weights
                    this.config.xAxis = Object.keys(result.nash_weights);
                    this.config.data = Object.values(result.nash_weights);
                } else if (result.shapley_values && typeof result.shapley_values === 'object') {
                    // Game theory - Shapley values
                    this.config.xAxis = Object.keys(result.shapley_values);
                    this.config.data = Object.values(result.shapley_values);
                } else if (result.optimal_bids && typeof result.optimal_bids === 'object') {
                    // Game theory - Auction bids
                    this.config.xAxis = Object.keys(result.optimal_bids);
                    this.config.data = Object.values(result.optimal_bids);
                } else if (result.scatter_points && Array.isArray(result.scatter_points)) {
                    // Color period regression format
                    this.config.scatter_points = result.scatter_points;
                    this.config.fitted_curve = result.fitted_curve || [];
                    this.config.isScatter = true;
                    this.config.r2 = result.r2 || 0;
                    this.config.coefficients = result.coefficients || {};
                } else if (result.points && Array.isArray(result.points) && result.cluster_labels) {
                    // Cluster visualization format
                    this.config.cluster_points = result.points;
                    this.config.cluster_labels = result.cluster_labels;
                    this.config.cluster_centers = result.cluster_centers || [];
                    this.config.n_clusters = result.n_clusters || 0;
                    this.config.isScatter = true;
                    this.config.isCluster = true;
                }
            } catch (error) {
                console.error('Failed to fetch bar chart data:', error);
                // Still render with empty/default data
            }
        }

        // Ensure we have at least empty arrays for rendering
        if (!this.config.xAxis) this.config.xAxis = [];
        if (!this.config.data) this.config.data = [];
        if (!this.config.series) this.config.series = [];

        this.render();
        
        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Render the bar chart using ECharts
     */
    render() {
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            container = document.querySelector(`#${this.containerId}-container`);
        }
        if (!container) return;

        if (!this.chart) {
            this.chart = echarts.init(container);
        }

        if (this.config.isScatter) {
            // Handle cluster visualization
            if (this.config.isCluster && this.config.cluster_points) {
                const clusterPoints = this.config.cluster_points;
                const clusterLabels = this.config.cluster_labels || [];
                const clusterCenters = this.config.cluster_centers || [];
                const nClusters = this.config.n_clusters || 0;
                
                // Color palette for clusters
                const isFinance = document.body.getAttribute('data-mode') === 'finance';
                const colors = isFinance ? 
                    ['#00ff88', '#00d4aa', '#00b8cc', '#009ce8', '#0080ff', '#00c4ff', '#00e8d4', '#00ffb8', '#00ff9c', '#00ff80'] :
                    ['#00ffff', '#00d4ff', '#00a8ff', '#007cff', '#0050ff', '#00ffd4', '#00ffa8', '#00ff7c', '#00ff50', '#00ff24'];
                
                // Group points by cluster
                const clusterSeries = [];
                for (let i = 0; i < nClusters; i++) {
                    const clusterData = [];
                    clusterPoints.forEach((point, idx) => {
                        if (clusterLabels[idx] === i) {
                            clusterData.push(point);
                        }
                    });
                    
                    if (clusterData.length > 0) {
                        clusterSeries.push({
                            name: `Cluster ${i}`,
                            type: 'scatter',
                            data: clusterData,
                            symbolSize: 8,
                            itemStyle: {
                                color: colors[i % colors.length],
                                opacity: 0.7
                            }
                        });
                    }
                }
                
                // Add cluster centers
                if (clusterCenters.length > 0) {
                    clusterSeries.push({
                        name: 'Cluster Centers',
                        type: 'scatter',
                        data: clusterCenters,
                        symbolSize: 20,
                        symbol: 'diamond',
                        itemStyle: {
                            color: '#ff00ff',
                            borderColor: '#ffffff',
                            borderWidth: 2
                        },
                        label: {
                            show: true,
                            position: 'top',
                            formatter: (params) => `C${params.dataIndex}`,
                            color: '#ffffff',
                            fontSize: 12,
                            fontWeight: 'bold'
                        }
                    });
                }
                
                const option = {
                    tooltip: {
                        trigger: 'item',
                        formatter: (params) => {
                            if (params.seriesName === 'Cluster Centers') {
                                return `Cluster Center ${params.dataIndex}<br/>PC1: ${params.value[0].toFixed(3)}<br/>PC2: ${params.value[1].toFixed(3)}`;
                            }
                            return `${params.seriesName}<br/>PC1: ${params.value[0].toFixed(3)}<br/>PC2: ${params.value[1].toFixed(3)}`;
                        },
                        backgroundColor: 'rgba(0, 0, 0, 0.85)',
                        borderColor: 'var(--accent-primary)',
                        borderWidth: 1,
                        textStyle: { color: '#fff', fontSize: 12 }
                    },
                    legend: {
                        data: clusterSeries.map(s => s.name),
                        textStyle: { color: 'var(--text-primary)' },
                        top: 10
                    },
                    xAxis: {
                        type: 'value',
                        name: 'Principal Component 1',
                        nameLocation: 'middle',
                        nameGap: 30,
                        nameTextStyle: { color: 'var(--text-primary)' },
                        axisLine: { lineStyle: { color: 'var(--border-color)' } },
                        axisLabel: { color: 'var(--text-secondary)' }
                    },
                    yAxis: {
                        type: 'value',
                        name: 'Principal Component 2',
                        nameLocation: 'middle',
                        nameGap: 50,
                        nameTextStyle: { color: 'var(--text-primary)' },
                        axisLine: { lineStyle: { color: 'var(--border-color)' } },
                        axisLabel: { color: 'var(--text-secondary)' }
                    },
                    series: clusterSeries,
                    grid: {
                        left: '10%',
                        right: '10%',
                        top: '15%',
                        bottom: '15%',
                        containLabel: true
                    }
                };
                
                if (window.GraphEnhancer) {
                    option = GraphEnhancer.enhanceOption('scatter', option, {
                        title: this.config.title,
                        stats: { n_clusters: nClusters }
                    });
                }
                
                this.chart.setOption(option, true);
                setTimeout(() => this.chart?.resize(), 100);
                return;
            }
            
            // Handle regular scatter plots (color-period regression, etc.)
            let scatterData = [];
            let fittedCurve = [];
            
            if (this.config.scatterData) {
                scatterData = this.config.scatterData.map(item => item.value || [item.value[0], item.value[1]]);
            } else if (this.config.scatter_points) {
                scatterData = this.config.scatter_points;
            }
            
            if (this.config.fitted_curve) {
                fittedCurve = this.config.fitted_curve;
            }
            
            let option = {
                tooltip: {
                    trigger: 'item',
                    formatter: (params) => {
                        if (params.seriesName === 'Fitted Curve') {
                            return `Fitted: ${params.value[0].toFixed(2)}, ${params.value[1].toFixed(2)}`;
                        }
                        return `Color Index: ${params.value[0].toFixed(3)}<br/>Rotation Period: ${params.value[1].toFixed(2)} days`;
                    }
                },
                xAxis: {
                    type: 'value',
                    name: 'Color Index (B-V or BP-RP)'
                },
                yAxis: {
                    type: 'value',
                    name: 'Rotation Period (days)'
                },
                legend: {
                    data: fittedCurve.length > 0 ? ['Data Points', 'Fitted Curve'] : ['Data Points'],
                    textStyle: {
                        color: 'var(--text-primary)'
                    }
                },
                series: [
                    {
                        name: 'Data Points',
                        type: 'scatter',
                        data: scatterData,
                        symbolSize: 8,
                        itemStyle: {
                            color: '#00ffff',
                            opacity: 0.7
                        }
                    }
                ]
            };
            
            if (fittedCurve.length > 0) {
                option.series.push({
                    name: 'Fitted Curve',
                    type: 'line',
                    data: fittedCurve,
                    smooth: true,
                    lineStyle: {
                        color: '#ff00ff',
                        width: 2
                    },
                    symbol: 'none',
                    itemStyle: {
                        color: '#ff00ff'
                    }
                });
            }
            if (window.GraphEnhancer) {
                option = GraphEnhancer.enhanceOption('bar', option, {
                    title: this.config.title,
                    xUnit: 'Color Index',
                    yUnit: 'Rotation Period (days)',
                    stats: { r2: this.config.r2, referenceLine: fittedCurve }
                });
            }
            
            this.chart.setOption(option, true);
            setTimeout(() => this.chart?.resize(), 100);
            return;
        }

        let option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: { type: 'shadow' }
            },
            xAxis: {
                type: 'category',
                data: this.config.xAxis || []
            },
            yAxis: {
                type: 'value'
            },
            series: [{
                name: this.config.title,
                type: 'bar',
                data: this.config.data || [],
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: '#83bff6' },
                        { offset: 0.5, color: '#188df0' },
                        { offset: 1, color: '#188df0' }
                    ])
                }
            }]
        };

        if (window.GraphEnhancer) {
            option = GraphEnhancer.enhanceOption('bar', option, {
                title: this.config.title,
                yUnit: this.config.unit || '',
                stats: this.config.stats
            });
        }

        this.chart.setOption(option, true);
        setTimeout(() => this.chart?.resize(), 100);
    }

    /**
     * Update chart with new data
     */
    update(data) {
        this.config = { ...this.config, ...data };
        if (this.chart) {
            const option = {};
            if (data.xAxis) {
                option.xAxis = { data: data.xAxis };
            }
            if (data.series) {
                option.series = data.series;
            } else if (data.data) {
                option.series = [{ data: data.data }];
            } else if (data.scatterData) {
                // Handle scatter plot update
                this.config.isScatter = true;
                this.config.scatterData = data.scatterData;
                this.render();
                return;
            }
            this.chart.setOption(option, true);  // true = notMerge for full update
        }
    }

    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = BarChart;
}
