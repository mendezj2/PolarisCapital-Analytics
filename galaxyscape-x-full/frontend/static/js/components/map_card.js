/**
 * Map Card Component
 * Displays geographic visualizations using ECharts geo maps
 * TODO (USER): Implement map visualization with ECharts
 */

class MapCard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Geographic Map',
            data: config.data || [],
            mapType: config.mapType || 'world', // world, china, usa, etc.
            apiEndpoint: config.apiEndpoint || null,
            ...config
        };
        this.chart = null;
    }

    /**
     * Initialize the map
     */
    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.warn(`Map container ${this.containerId} not found`);
            return;
        }

        // TODO (USER): Load map JSON data for ECharts
        // ECharts requires map JSON files to be loaded
        // Example: https://echarts.apache.org/examples/data/asset/geo/World.json

        // TODO (USER): Fetch data from API
        // if (this.config.apiEndpoint) {
        //     fetch(this.config.apiEndpoint)
        //         .then(res => res.json())
        //         .then(data => this.update(data));
        // }

        this.render();
    }

    /**
     * Render the map using ECharts
     * TODO (USER): Complete ECharts geo map configuration
     * Reference: https://echarts.apache.org/en/option.html#geo
     */
    render() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        if (!this.chart) {
            this.chart = echarts.init(container);
        }

        // TODO (USER): Configure ECharts geo map option
        // Note: Map JSON must be loaded first using echarts.registerMap()
        const option = {
            tooltip: {
                trigger: 'item',
                formatter: '{b}: {c}'
            },
            visualMap: {
                min: 0,
                max: 1000,
                left: 'left',
                top: 'bottom',
                text: ['High', 'Low'],
                calculable: true,
                inRange: {
                    color: ['#e0f3ff', '#0066cc']
                }
            },
            geo: {
                map: this.config.mapType,
                roam: true,
                emphasis: {
                    label: {
                        show: true
                    }
                }
            },
            series: [{
                name: 'Data',
                type: 'map',
                map: this.config.mapType,
                data: this.config.data
            }]
        };

        // TODO (USER): Load map JSON before setting option
        // echarts.registerMap('world', worldJson);
        // this.chart.setOption(option);
    }

    /**
     * Update map with new data
     */
    update(data) {
        this.config = { ...this.config, ...data };
        if (this.chart) {
            this.chart.setOption({
                series: [{ data: this.config.data }]
            });
        }
    }

    resize() {
        if (this.chart) {
            this.chart.resize();
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = MapCard;
}

