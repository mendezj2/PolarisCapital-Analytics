/**
 * Line Chart Component
 * Displays time series and line charts using ECharts
 * TODO (USER): Implement line chart visualization with ECharts
 */

class LineChart {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Line Chart',
            xAxis: config.xAxis || [],
            series: config.series || [],
            apiEndpoint: config.apiEndpoint || null,
            ...config
        };
        this.chart = null;
    }

    /**
     * Initialize the line chart
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
            console.warn(`Line chart container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API if endpoint provided
        if (this.config.apiEndpoint) {
            try {
                // Check if this is a POST endpoint (stock/compare or future/outcomes)
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
                } else if (this.config.apiEndpoint.includes('/future/outcomes')) {
                    // Get portfolio from portfolio-input component
                    let portfolio = {};
                    if (typeof PortfolioInput !== 'undefined') {
                        const component = window.layoutManager?.components?.get('portfolio-input');
                        if (component && typeof component.getPortfolio === 'function') {
                            portfolio = component.getPortfolio();
                        }
                    }
                    
                    // If no portfolio, use default
                    if (Object.keys(portfolio).length === 0) {
                        portfolio = {'AAPL': 10, 'MSFT': 5};
                    }
                    
                    response = await fetch(this.config.apiEndpoint, {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({portfolio: portfolio})
                    });
                } else {
                    response = await fetch(this.config.apiEndpoint);
                }
                const result = await response.json();
                
                // Handle different response formats
                if (result.comparison && Array.isArray(result.comparison)) {
                    // Stock comparison - returns over time
                    this.config.series = [
                        {
                            name: '1 Month Return',
                            data: result.comparison.map(item => item.returns_1m),
                            type: 'line'
                        },
                        {
                            name: '3 Month Return',
                            data: result.comparison.map(item => item.returns_3m),
                            type: 'line'
                        },
                        {
                            name: '1 Year Return',
                            data: result.comparison.map(item => item.returns_1y),
                            type: 'line'
                        }
                    ];
                    this.config.xAxis = result.comparison.map(item => item.ticker);
                } else if (result.xAxis && result.series) {
                    this.config.xAxis = result.xAxis;
                    this.config.series = result.series;
                } else if (result.xAxis && result.data) {
                    this.config.xAxis = result.xAxis;
                    this.config.series = [{
                        name: this.config.title,
                        data: result.data
                    }];
                } else if (result.actual && Array.isArray(result.actual)) {
                    // Regression plot format (ML models)
                    this.config.series = [
                        {
                            name: 'Actual',
                            data: result.actual,
                            type: 'line'
                        },
                        {
                            name: 'Predicted',
                            data: result.predicted || [],
                            type: 'line'
                        }
                    ];
                    this.config.xAxis = result.actual.map((_, i) => i);
                } else if (result.residuals && Array.isArray(result.residuals)) {
                    // Residuals plot format
                    this.config.series = [{
                        name: 'Residuals',
                        data: result.residuals,
                        type: 'line'
                    }];
                    this.config.xAxis = result.residuals.map((_, i) => i);
                } else if (result.strategy_shares && Array.isArray(result.strategy_shares)) {
                    // Game theory - Evolutionary dynamics
                    const tickers = Object.keys(result.final_shares || {});
                    this.config.xAxis = result.strategy_shares.map((_, i) => i);
                    this.config.series = tickers.map((ticker, idx) => ({
                        name: ticker,
                        data: result.strategy_shares.map(shares => shares[idx] || 0),
                        type: 'line',
                        smooth: true
                    }));
                } else if (result.channels && Array.isArray(result.channels)) {
                    // Omni-channel format
                    this.config.xAxis = result.channels.map(item => item.channel);
                    this.config.series = [
                        {
                            name: 'Messaging Consistency',
                            data: result.channels.map(item => item.messaging_consistency),
                            type: 'line'
                        },
                        {
                            name: 'Customer Engagement',
                            data: result.channels.map(item => item.customer_engagement),
                            type: 'line'
                        }
                    ];
                } else if (result.journey_stages && Array.isArray(result.journey_stages)) {
                    // PRO Consumer Journey format
                    this.config.xAxis = result.journey_stages.map(item => item.stage);
                    this.config.series = [{
                        name: 'PRO Customers',
                        data: result.journey_stages.map(item => item.pro_customers),
                        type: 'line'
                    }];
                }
            } catch (error) {
                console.error('Failed to fetch line chart data:', error);
                // Still render with empty/default data
            }
        }

        // Ensure we have at least empty arrays for rendering
        if (!this.config.xAxis) this.config.xAxis = [];
        if (!this.config.series) this.config.series = [];

        this.render();
        
        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Render the line chart using ECharts
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

        // TODO (USER): Configure ECharts line chart option
        // Reference: https://echarts.apache.org/en/option.html#series.line
        let option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'var(--bg-card)',
                borderColor: 'var(--border-color)',
                textStyle: {
                    color: 'var(--text-primary)',
                    fontSize: 12
                }
            },
            legend: {
                data: this.config.series.map(s => s.name)
            },
            grid: {
                left: '3%',
                right: '4%',
                bottom: '3%',
                containLabel: true
            },
            xAxis: {
                type: 'category',
                boundaryGap: false,
                data: this.config.xAxis || [],
                axisLabel: {
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    fontWeight: 500
                }
            },
            yAxis: {
                type: 'value',
                axisLabel: {
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    fontWeight: 500
                }
            },
            series: (this.config.series || []).map(s => ({
                name: s.name,
                type: 'line',
                smooth: true,
                data: s.data || [],
                areaStyle: s.areaStyle || undefined
            }))
        };

        // Ensure chart renders even with empty data
        if ((this.config.xAxis || []).length === 0 && (this.config.series || []).length === 0) {
            // Show placeholder message
            option.graphic = [{
                type: 'text',
                left: 'center',
                top: 'middle',
                style: {
                    text: 'No data available',
                    fontSize: 14,
                    fill: 'var(--text-secondary)'
                }
            }];
        }
        
        if (window.GraphEnhancer) {
            option = GraphEnhancer.enhanceOption('line', option, {
                title: this.config.title,
                xUnit: this.config.xUnit,
                yUnit: this.config.yUnit,
                stats: this.config.stats
            });
        }

        this.chart.setOption(option, true);
        
        // Resize chart to ensure proper rendering
        setTimeout(() => {
            if (this.chart) {
                this.chart.resize();
            }
        }, 100);
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
    module.exports = LineChart;
}
