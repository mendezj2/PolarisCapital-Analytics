/**
 * Enhanced Pie Chart Component
 * Displays beautiful pie/donut charts using ECharts with improved styling
 */

class PieChart {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Pie Chart',
            data: config.data || [],
            apiEndpoint: config.apiEndpoint || null,
            showLegend: config.showLegend !== false,
            showLabels: config.showLabels !== true, // Default to false for cleaner look
            donut: config.donut !== false, // Default to donut chart
            colors: config.colors || null, // Custom color palette
            ...config
        };
        this.chart = null;
    }

    /**
     * Initialize the pie chart
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
            console.warn(`Pie chart container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API
        if (this.config.apiEndpoint) {
            try {
                const response = await fetch(this.config.apiEndpoint);
                const result = await response.json();
                
                // Handle different response formats
                if (result.sector_breakdown && Array.isArray(result.sector_breakdown)) {
                    // Stock explorer - sector breakdown (from API)
                    this.config.data = result.sector_breakdown.map(item => ({
                        name: item.sector || item.name,
                        value: item.count || item.value
                    }));
                } else if (result.stocks && Array.isArray(result.stocks)) {
                    // Stock explorer - sector breakdown (calculate from stocks)
                    const sectorCounts = {};
                    result.stocks.forEach(stock => {
                        const sector = stock.sector || 'Unknown';
                        sectorCounts[sector] = (sectorCounts[sector] || 0) + 1;
                    });
                    this.config.data = Object.entries(sectorCounts).map(([name, value]) => ({
                        name,
                        value
                    }));
                } else if (result.data && Array.isArray(result.data)) {
                    this.config.data = result.data;
                } else if (result.allocation && result.allocation.data) {
                    this.config.data = result.allocation.data;
                } else if (result.current_allocation && Array.isArray(result.current_allocation)) {
                    // Expenditure optimization
                    const isOptimized = this.containerId.includes('optimized');
                    this.config.data = result.current_allocation.map(item => ({
                        name: item.channel,
                        value: isOptimized ? (item.recommended_budget || item.budget) : item.budget
                    }));
                }
            } catch (error) {
                console.error('Failed to fetch pie chart data:', error);
            }
        }

        // Ensure we have at least empty array for rendering
        if (!this.config.data) this.config.data = [];
        
        // Validate data format
        if (this.config.data.length > 0) {
            this.config.data = this.config.data.filter(item => {
                if (typeof item === 'object' && item !== null) {
                    return item.hasOwnProperty('name') && item.hasOwnProperty('value');
                }
                return false;
            });
        }

        this.render();
        
        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Get color palette based on theme
     */
    getColorPalette() {
        if (this.config.colors && Array.isArray(this.config.colors)) {
            return this.config.colors;
        }

        // Default vibrant color palettes
        const isFinance = document.body.getAttribute('data-mode') === 'finance';
        
        if (isFinance) {
            // Finance theme - green/teal palette
            return [
                '#00ff88', '#00d4aa', '#00b8cc', '#009ce8', '#0080ff',
                '#00c4ff', '#00e8d4', '#00ffb8', '#00ff9c', '#00ff80',
                '#4dffaa', '#66ffb8', '#80ffc6', '#99ffd4', '#b3ffe2'
            ];
        } else {
            // Astronomy theme - cyan/blue palette
            return [
                '#00ffff', '#00d4ff', '#00a8ff', '#007cff', '#0050ff',
                '#00ffd4', '#00ffa8', '#00ff7c', '#00ff50', '#00ff24',
                '#24ffff', '#48ffff', '#6cffff', '#90ffff', '#b4ffff'
            ];
        }
    }

    /**
     * Render the pie chart using ECharts with enhanced styling
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

        // Ensure data is valid
        if (!this.config.data || this.config.data.length === 0) {
            container.innerHTML = '<div class="chart-empty-state"><span class="empty-icon">📊</span><p>No data available</p></div>';
            return;
        }
        
        // Process dataset
        const dataset = (this.config.data || []).map((item, index) => ({
            name: item.name || `Item ${index + 1}`,
            value: typeof item.value === 'number' ? item.value : parseFloat(item.value) || 0
        })).filter(item => item.value > 0);

        if (dataset.length === 0) {
            container.innerHTML = '<div class="chart-empty-state"><span class="empty-icon">📊</span><p>No valid data</p></div>';
            return;
        }

        const total = dataset.reduce((sum, d) => sum + (d.value || 0), 0);
        const colorPalette = this.getColorPalette();

        // Enhanced ECharts option
        let option = {
            backgroundColor: 'transparent',
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(0, 0, 0, 0.85)',
                borderColor: 'var(--accent-primary)',
                borderWidth: 1,
                textStyle: {
                    color: '#fff',
                    fontSize: 13,
                    fontWeight: 500
                },
                padding: [10, 15],
                formatter: (params) => {
                    const value = typeof params.value === 'number' ? params.value.toLocaleString() : params.value;
                    const percent = (params.percent || 0).toFixed(1);
                    return `
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="display: inline-block; width: 12px; height: 12px; background: ${params.color}; border-radius: 2px;"></span>
                            <strong>${params.name}</strong>
                        </div>
                        <div style="margin-top: 6px; font-size: 14px;">
                            <div>Value: <strong>${value}</strong></div>
                            <div>Percentage: <strong>${percent}%</strong></div>
                        </div>
                    `;
                },
                extraCssText: 'box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3); border-radius: 4px;'
            },
            legend: this.config.showLegend !== false ? {
                orient: this.config.legendOrientation || 'vertical',
                top: 'middle',
                right: 20,
                left: 'auto',
                itemWidth: 14,
                itemHeight: 14,
                itemGap: 12,
                textStyle: {
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    fontWeight: 500,
                    lineHeight: 18
                },
                formatter: (name) => {
                    const item = dataset.find(d => d.name === name);
                    if (item && typeof item.value === 'number' && total > 0) {
                        const percent = (item.value / total * 100).toFixed(1);
                        return `${name}  ${item.value} (${percent}%)`;
                    }
                    return name;
                },
                selectedMode: false
            } : undefined,
            series: [{
                name: this.config.title || 'Data',
                type: 'pie',
                radius: this.config.donut !== false ? ['45%', '75%'] : ['0%', '75%'], // Donut chart
                center: this.config.showLegend !== false ? ['40%', '50%'] : ['50%', '50%'],
                avoidLabelOverlap: true,
                itemStyle: {
                    borderRadius: 8,
                    borderColor: 'var(--bg-primary)',
                    borderWidth: 3,
                    shadowBlur: 0,
                    shadowColor: 'transparent'
                },
                label: {
                    show: this.config.showLabels === true,
                    position: 'outside',
                    alignTo: 'edge',
                    margin: 8,
                    formatter: (params) => {
                        const percent = (params.percent || 0).toFixed(1);
                        return `${params.name}\n${percent}%`;
                    },
                    color: 'var(--text-primary)',
                    fontSize: 11,
                    fontWeight: 600,
                    lineHeight: 16,
                    rich: {
                        name: {
                            fontSize: 11,
                            fontWeight: 600,
                            color: 'var(--text-primary)'
                        },
                        percent: {
                            fontSize: 10,
                            fontWeight: 500,
                            color: 'var(--text-secondary)'
                        }
                    }
                },
                labelLine: {
                    show: this.config.showLabels === true,
                    length: 15,
                    length2: 10,
                    smooth: 0.2,
                    lineStyle: {
                        color: 'var(--text-secondary)',
                        width: 1.5,
                        type: 'solid'
                    }
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: 13,
                        fontWeight: 'bold',
                        color: 'var(--text-primary)'
                    },
                    itemStyle: {
                        shadowBlur: 20,
                        shadowOffsetX: 0,
                        shadowOffsetY: 0,
                        shadowColor: 'rgba(0, 255, 136, 0.5)',
                        borderWidth: 4,
                        scale: true,
                        scaleSize: 5
                    },
                    focus: 'self'
                },
                data: dataset.map((item, index) => ({
                    ...item,
                    itemStyle: {
                        color: colorPalette[index % colorPalette.length]
                    }
                }))
            }]
        };

        // Apply theme-specific enhancements
        if (window.GraphEnhancer) {
            option = GraphEnhancer.enhanceOption('pie', option, {
                title: this.config.title,
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
        
        // Additional resize on window resize
        if (!this.resizeHandler) {
            this.resizeHandler = () => {
                if (this.chart) {
                    this.chart.resize();
                }
            };
            window.addEventListener('resize', this.resizeHandler);
        }
    }

    /**
     * Update chart with new data
     */
    update(data) {
        if (data.data && Array.isArray(data.data)) {
            this.config.data = data.data;
        } else if (Array.isArray(data)) {
            this.config.data = data;
        } else {
            this.config = { ...this.config, ...data };
        }
        
        // Validate data format
        if (this.config.data && this.config.data.length > 0) {
            this.config.data = this.config.data.filter(item => {
                if (typeof item === 'object' && item !== null) {
                    return item.hasOwnProperty('name') && item.hasOwnProperty('value');
                }
                return false;
            }).map((item, index) => ({
                name: item.name || `Item ${index + 1}`,
                value: typeof item.value === 'number' ? item.value : parseFloat(item.value) || 0
            })).filter(item => item.value > 0);
        }
        
        // Re-render with updated data
        this.render();
    }

    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }

    destroy() {
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PieChart;
}
