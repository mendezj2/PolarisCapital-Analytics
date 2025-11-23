/**
 * Gauge Card Component
 * Displays gauge/radial charts using ECharts
 * TODO (USER): Implement gauge visualization with ECharts
 */

class GaugeCard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Gauge',
            value: config.value || 0,
            min: config.min || 0,
            max: config.max || 100,
            thresholds: config.thresholds || [
                { value: 0, color: '#ff6b6b' },
                { value: 50, color: '#ffd43b' },
                { value: 80, color: '#51cf66' }
            ],
            apiEndpoint: config.apiEndpoint || null,
            ...config
        };
        this.chart = null;
    }

    /**
     * Initialize the gauge chart
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
            console.warn(`Gauge container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API if apiEndpoint is provided
        if (this.config.apiEndpoint) {
            try {
                const response = await fetch(this.config.apiEndpoint);
                const result = await response.json();
                
                // Handle different response formats
                if (result.value !== undefined) {
                    this.config.value = result.value;
                    this.config.min = result.min || this.config.min;
                    this.config.max = result.max || this.config.max;
                } else if (result.risk_score !== undefined) {
                    this.config.value = result.risk_score;
                    this.config.min = 0;
                    this.config.max = 100;
                } else if (result.risk_level !== undefined) {
                    this.config.value = result.risk_level.value || result.risk_level;
                    this.config.min = result.risk_level.min || 0;
                    this.config.max = result.risk_level.max || 100;
                }
            } catch (error) {
                console.error('Failed to fetch gauge data:', error);
            }
        }

        this.render();
        
        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Render the gauge chart using ECharts
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

        // Initialize ECharts instance
        if (!this.chart) {
            this.chart = echarts.init(container);
        }

        // TODO (USER): Configure ECharts gauge option
        // Reference: https://echarts.apache.org/en/option.html#series.gauge
        const option = {
            series: [{
                type: 'gauge',
                startAngle: 180,
                endAngle: 0,
                min: this.config.min,
                max: this.config.max,
                splitNumber: 8,
                axisLine: {
                    lineStyle: {
                        width: 6,
                        color: this.config.thresholds.map(t => [t.value / this.config.max, t.color])
                    }
                },
                pointer: {
                    itemStyle: {
                        color: 'auto'
                    }
                },
                axisTick: {
                    distance: -30,
                    length: 8,
                    lineStyle: {
                        color: '#fff',
                        width: 2
                    }
                },
                splitLine: {
                    distance: -30,
                    length: 14,
                    lineStyle: {
                        color: '#fff',
                        width: 2
                    }
                },
                axisLabel: {
                    distance: -20,
                    color: 'var(--text-primary)',
                    fontSize: 12,
                    fontWeight: 500
                },
                detail: {
                    valueAnimation: true,
                    formatter: '{value}',
                    color: 'var(--text-primary)',
                    fontSize: 24,
                    fontWeight: 'bold'
                },
                data: [{
                    value: this.config.value,
                    name: this.config.title
                }]
            }]
        };

        this.chart.setOption(option);
        
        // Resize chart to ensure proper rendering
        setTimeout(() => {
            if (this.chart) {
                this.chart.resize();
            }
        }, 100);
    }

    destroy() {
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }

    /**
     * Update gauge with new value
     */
    update(data) {
        this.config = { ...this.config, ...data };
        if (this.chart) {
            this.chart.setOption({
                series: [{
                    data: [{
                        value: this.config.value,
                        name: this.config.title
                    }]
                }]
            });
        }
    }

    /**
     * Resize chart (call on window resize)
     */
    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GaugeCard;
}

