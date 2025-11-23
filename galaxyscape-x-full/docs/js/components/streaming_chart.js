/**
 * Streaming Chart Component
 * Displays real-time updating charts for streaming data
 */

class StreamingChart {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Streaming Chart',
            apiEndpoint: config.apiEndpoint || null,
            updateInterval: config.updateInterval || 5000, // 5 seconds
            maxDataPoints: config.maxDataPoints || 100,
            ...config
        };
        this.chart = null;
        this.dataBuffer = [];
        this.updateTimer = null;
    }

    /**
     * Initialize the streaming chart
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
            console.warn(`Streaming chart container ${this.containerId} not found`);
            return;
        }

        if (!this.chart) {
            this.chart = echarts.init(container);
        }

        // Initial data load
        await this.fetchAndUpdate();

        // Start streaming updates
        this.startStreaming();

        // Add resize handler
        window.addEventListener('resize', () => this.resize());
    }

    /**
     * Start streaming updates
     */
    startStreaming() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
        }

        this.updateTimer = setInterval(() => {
            this.fetchAndUpdate();
        }, this.config.updateInterval);
    }

    /**
     * Stop streaming updates
     */
    stopStreaming() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    /**
     * Fetch and update chart data
     */
    async fetchAndUpdate() {
        if (!this.config.apiEndpoint) return;

        try {
            const response = await fetch(this.config.apiEndpoint);
            const result = await response.json();

            if (result.risk_scores) {
                // Handle risk scores format
                const now = new Date();
                result.risk_scores.forEach(score => {
                    this.dataBuffer.push({
                        time: now.toLocaleTimeString(),
                        value: score.risk_score,
                        ticker: score.ticker
                    });
                });
            } else if (result.series) {
                // Handle time series format
                if (result.series[0] && result.series[0].data) {
                    result.series[0].data.forEach((value, idx) => {
                        this.dataBuffer.push({
                            time: result.xAxis[idx] || new Date().toLocaleTimeString(),
                            value: value
                        });
                    });
                }
            }

            // Limit buffer size
            if (this.dataBuffer.length > this.config.maxDataPoints) {
                this.dataBuffer = this.dataBuffer.slice(-this.config.maxDataPoints);
            }

            this.render();
        } catch (error) {
            console.error('Failed to fetch streaming data:', error);
        }
    }

    /**
     * Render the streaming chart
     */
    render() {
        if (!this.chart) return;

        const times = this.dataBuffer.map(d => d.time);
        const values = this.dataBuffer.map(d => d.value);

        let option = {
            tooltip: {
                trigger: 'axis',
                axisPointer: {
                    type: 'cross'
                }
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
                data: times
            },
            yAxis: {
                type: 'value'
            },
            series: [{
                name: this.config.title,
                type: 'line',
                smooth: true,
                data: values,
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(34, 197, 94, 0.3)' },
                        { offset: 1, color: 'rgba(34, 197, 94, 0.1)' }
                    ])
                },
                lineStyle: {
                    color: '#22c55e',
                    width: 2
                }
            }],
            animation: true,
            animationDuration: 300
        };

        if (window.GraphEnhancer) {
            option = GraphEnhancer.enhanceOption('line', option, {
                title: this.config.title,
                yUnit: 'Risk Score'
            });
        }

        this.chart.setOption(option, true);
    }

    /**
     * Update chart with new data
     */
    update(data) {
        if (data.risk_scores) {
            data.risk_scores.forEach(score => {
                this.dataBuffer.push({
                    time: new Date().toLocaleTimeString(),
                    value: score.risk_score
                });
            });
        }
        if (this.dataBuffer.length > this.config.maxDataPoints) {
            this.dataBuffer = this.dataBuffer.slice(-this.config.maxDataPoints);
        }
        this.render();
    }

    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }

    destroy() {
        this.stopStreaming();
        if (this.chart) {
            this.chart.dispose();
            this.chart = null;
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = StreamingChart;
}



