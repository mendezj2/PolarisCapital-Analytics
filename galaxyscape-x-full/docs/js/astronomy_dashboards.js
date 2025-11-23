/**
 * Astronomy Dashboards Configuration
 * Defines dashboard layouts and component configurations for Astronomy domain
 * TODO (USER): Implement dashboard-specific data loading and component initialization
 */

const AstronomyDashboards = {
    /**
     * Overview Dashboard Configuration
     */
    overview: {
        layout: 'grid',
        requiredFeatures: ['age', 'temperature', 'cluster'],
        components: [
            {
                type: 'kpi',
                id: 'kpi-total-stars',
                title: 'Total Stars',
                apiEndpoint: '/api/astronomy/dashboard/kpi?metric=total_stars',
                position: { row: 1, col: 1 }
            },
            {
                type: 'kpi',
                id: 'kpi-avg-age',
                title: 'Average Age',
                apiEndpoint: '/api/astronomy/dashboard/kpi?metric=avg_age',
                position: { row: 1, col: 2 }
            },
            {
                type: 'line-chart',
                id: 'chart-age-distribution',
                title: 'Age Distribution Over Time',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=age',
                position: { row: 2, col: 1, span: 2 }
            },
            {
                type: 'bar-chart',
                id: 'chart-cluster-sizes',
                title: 'Cluster Sizes',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=clusters',
                position: { row: 3, col: 1 }
            },
            {
                type: 'pie-chart',
                id: 'chart-cluster-dist',
                title: 'Cluster Distribution',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=cluster_dist',
                position: { row: 3, col: 2 }
            }
        ]
    },

    /**
     * Star Age Dashboard Configuration
     */
    'star-age': {
        layout: 'grid',
        requiredFeatures: ['age'],
        components: [
            {
                type: 'gauge',
                id: 'gauge-data-quality',
                title: 'Data Quality Score',
                apiEndpoint: '/api/astronomy/dashboard/kpi?metric=data_quality',
                position: { row: 1, col: 1 }
            },
            {
                type: 'line-chart',
                id: 'chart-age-trends',
                title: 'Age Prediction Trends',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=age_predictions',
                position: { row: 1, col: 2, span: 2 }
            },
            {
                type: 'data-table',
                id: 'table-age-predictions',
                title: 'Age Predictions',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=age_table',
                position: { row: 2, col: 1, span: 2 }
            }
        ]
    },


    /**
     * Sky Network Dashboard Configuration
     */
    'sky-network': {
        layout: 'grid',
        requiredFeatures: ['cluster', 'temperature'],
        components: [
            {
                type: 'network-graph',
                id: 'network-sky',
                title: 'Stellar Network Graph',
                apiEndpoint: '/api/astronomy/dashboard/network?type=stellar',
                position: { row: 1, col: 1, span: 2, fullWidth: true }
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-stars',
                title: 'Most Connected Stars',
                apiEndpoint: '/api/astronomy/dashboard/leaderboard?metric=connectivity',
                position: { row: 2, col: 1 }
            },
            {
                type: 'kpi',
                id: 'kpi-network-metrics',
                title: 'Network Metrics',
                apiEndpoint: '/api/astronomy/dashboard/kpi?metric=network',
                position: { row: 2, col: 2 }
            }
        ]
    },

    /**
     * Star Explorer Dashboard Configuration
     */
    'star-explorer': {
        layout: 'grid',
        requiredFeatures: ['rotation_period', 'color_index', 'mass'],
        globalFilters: [
            {
                name: 'search',
                label: 'Search Star',
                type: 'search',
                placeholder: 'Search by name or ID...',
                debounce: 300
            },
            {
                name: 'rotation_min',
                label: 'Rotation Period (Min)',
                type: 'number',
                defaultMin: 0
            },
            {
                name: 'rotation_max',
                label: 'Rotation Period (Max)',
                type: 'number',
                defaultMax: 100
            },
            {
                name: 'color_min',
                label: 'Color Index (Min)',
                type: 'number',
                defaultMin: -1
            },
            {
                name: 'color_max',
                label: 'Color Index (Max)',
                type: 'number',
                defaultMax: 2
            },
            {
                name: 'mass_min',
                label: 'Mass (Min)',
                type: 'number',
                defaultMin: 0
            },
            {
                name: 'mass_max',
                label: 'Mass (Max)',
                type: 'number',
                defaultMax: 100
            },
            {
                name: 'metallicity_min',
                label: 'Metallicity (Min)',
                type: 'number',
                defaultMin: -1
            },
            {
                name: 'metallicity_max',
                label: 'Metallicity (Max)',
                type: 'number',
                defaultMax: 1
            },
            {
                name: 'temperature_min',
                label: 'Temperature (Min)',
                type: 'number',
                defaultMin: 0
            },
            {
                name: 'temperature_max',
                label: 'Temperature (Max)',
                type: 'number',
                defaultMax: 50000
            },
            {
                name: 'age_min',
                label: 'Age (Min) Gyr',
                type: 'number',
                defaultMin: 0
            },
            {
                name: 'age_max',
                label: 'Age (Max) Gyr',
                type: 'number',
                defaultMax: 15
            },
            {
                name: 'clusters',
                label: 'Clusters',
                type: 'multi-select',
                options: [],  // Will be populated from data
                defaultValue: []
            },
            {
                name: 'anomalies_only',
                label: 'Show Anomalies Only',
                type: 'toggle',
                onLabel: 'Anomalies',
                offLabel: 'All',
                defaultValue: false
            }
        ],
        components: [
            {
                type: 'data-table',
                id: 'table-stars',
                title: 'Star Catalog',
                apiEndpoint: '/api/astronomy/star_table',
                position: { row: 1, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-star-scatter',
                title: 'Color Index vs Rotation Period',
                apiEndpoint: '/api/astronomy/color_period',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true,
                isScatter: true
            }
        ]
    },

    /**
     * Sky Map Dashboard Configuration
     */
    'sky-map': {
        layout: 'grid',
        requiredFeatures: ['ra', 'dec'],
        globalFilters: [
            {
                name: 'search',
                label: 'Search Star',
                type: 'search',
                placeholder: 'Search by name...',
                debounce: 300
            },
            {
                name: 'clusters',
                label: 'Clusters',
                type: 'multi-select',
                options: [],
                defaultValue: []
            },
            {
                name: 'temperature_min',
                label: 'Min Temperature',
                type: 'slider',
                min: 0,
                max: 50000,
                step: 100,
                defaultValue: 0
            },
            {
                name: 'temperature_max',
                label: 'Max Temperature',
                type: 'slider',
                min: 0,
                max: 50000,
                step: 100,
                defaultValue: 50000
            },
            {
                name: 'magnitude_min',
                label: 'Min Magnitude',
                type: 'slider',
                min: -5,
                max: 20,
                step: 0.1,
                defaultValue: -5
            },
            {
                name: 'magnitude_max',
                label: 'Max Magnitude',
                type: 'slider',
                min: -5,
                max: 20,
                step: 0.1,
                defaultValue: 20
            }
        ],
        components: [
            {
                type: 'bar-chart',
                id: 'chart-sky-map',
                title: 'Sky Map (RA vs Dec)',
                apiEndpoint: '/api/astronomy/sky_map',
                position: { row: 1, col: 1, span: 2, fullWidth: true },
                useFilteredData: true
            }
        ]
    },

    /**
     * Light Curve Dashboard Configuration
     */
    'light-curve': {
        layout: 'grid',
        requiredFeatures: ['magnitude', 'rotation_period'],
        globalFilters: [
            {
                name: 'star_name',
                label: 'Select Star',
                type: 'search',
                placeholder: 'Search star name...',
                debounce: 300
            },
            {
                name: 'star_id',
                label: 'Star ID',
                type: 'number',
                defaultMin: 0,
                defaultMax: 1000
            },
            {
                name: 'period_min',
                label: 'Min Period (days)',
                type: 'slider',
                min: 0.1,
                max: 100,
                step: 0.1,
                defaultValue: 0.1
            },
            {
                name: 'period_max',
                label: 'Max Period (days)',
                type: 'slider',
                min: 0.1,
                max: 100,
                step: 0.1,
                defaultValue: 100
            }
        ],
        components: [
            {
                type: 'line-chart',
                id: 'chart-light-curve',
                title: 'Light Curve',
                apiEndpoint: '/api/astronomy/light_curve',
                position: { row: 1, col: 1, span: 2 },
                useFilteredData: true
            }
        ]
    },
    
    /**
     * Cluster Dashboard Configuration
     */
    clusters: {
        layout: 'grid',
        requiredFeatures: ['cluster'],
        globalFilters: [
            {
                name: 'clusters',
                label: 'Select Clusters',
                type: 'multi-select',
                options: [],
                defaultValue: []
            },
            {
                name: 'cluster_size_min',
                label: 'Min Cluster Size',
                type: 'slider',
                min: 1,
                max: 100,
                step: 1,
                defaultValue: 1
            },
            {
                name: 'n_clusters',
                label: 'Number of Clusters',
                type: 'slider',
                min: 2,
                max: 10,
                step: 1,
                defaultValue: 5
            },
            {
                name: 'search',
                label: 'Search Star',
                type: 'search',
                placeholder: 'Search within clusters...',
                debounce: 300
            }
        ],
        components: [
            {
                type: 'bar-chart',
                id: 'chart-cluster-visualization',
                title: 'Cluster Visualization (2D Projection)',
                apiEndpoint: '/api/astronomy/cluster',
                position: { row: 1, col: 1, span: 2, fullWidth: true },
                isScatter: true,
                useFilteredData: false
            },
            {
                type: 'kpi',
                id: 'kpi-total-clusters',
                title: 'Total Clusters',
                apiEndpoint: '/api/astronomy/cluster',
                position: { row: 2, col: 1 },
                customRender: true
            },
            {
                type: 'data-table',
                id: 'table-cluster-stars',
                title: 'Stars by Cluster',
                apiEndpoint: '/api/astronomy/star_table',
                position: { row: 2, col: 2, span: 2 },
                useFilteredData: true
            }
        ]
    },
    
    /**
     * Anomaly Dashboard Configuration
     */
    anomalies: {
        layout: 'grid',
        requiredFeatures: ['temperature', 'mass'],
        globalFilters: [
            {
                name: 'anomalies_only',
                label: 'Show Anomalies Only',
                type: 'toggle',
                onLabel: 'Anomalies',
                offLabel: 'All',
                defaultValue: true
            },
            {
                name: 'anomaly_score_min',
                label: 'Min Anomaly Score',
                type: 'slider',
                min: 0,
                max: 1,
                step: 0.01,
                defaultValue: 0
            },
            {
                name: 'search',
                label: 'Search Star',
                type: 'search',
                placeholder: 'Search anomalies...',
                debounce: 300
            },
            {
                name: 'mass_min',
                label: 'Mass Range',
                type: 'number',
                defaultMin: 0,
                defaultMax: 100
            },
            {
                name: 'temperature_min',
                label: 'Temperature Range',
                type: 'number',
                defaultMin: 0,
                defaultMax: 50000
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-anomalies',
                title: 'Anomalies Detected',
                apiEndpoint: '/api/astronomy/dashboard/kpi?metric=anomalies',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-anomalies',
                title: 'Anomaly Details',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=anomaly_table',
                position: { row: 1, col: 2, span: 2 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-anomaly-trends',
                title: 'Anomaly Detection Over Time',
                apiEndpoint: '/api/astronomy/dashboard/trends?metric=anomaly_trends',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            }
        ]
    },

    /**
     * ML Models Dashboard Configuration
     */
    'ml-models': {
        layout: 'grid',
        requiredFeatures: ['age', 'temperature', 'mass', 'color_index'],
        globalFilters: [
            {
                name: 'model_type',
                label: 'Model Type',
                type: 'single-select',
                options: [
                    { value: 'xgboost', label: 'XGBoost' },
                    { value: 'lightgbm', label: 'LightGBM' },
                    { value: 'both', label: 'Compare Both' }
                ],
                defaultValue: 'both'
            },
            {
                name: 'metric',
                label: 'Performance Metric',
                type: 'single-select',
                options: [
                    { value: 'mae', label: 'Mean Absolute Error' },
                    { value: 'rmse', label: 'Root Mean Squared Error' },
                    { value: 'r2', label: 'R² Score' }
                ],
                defaultValue: 'r2'
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-xgb-mae',
                title: 'XGBoost MAE',
                apiEndpoint: '/api/astronomy/ml/models/performance',
                position: { row: 1, col: 1 },
                customRender: true
            },
            {
                type: 'kpi',
                id: 'kpi-lgbm-mae',
                title: 'LightGBM MAE',
                apiEndpoint: '/api/astronomy/ml/models/performance',
                position: { row: 1, col: 2 },
                customRender: true
            },
            {
                type: 'kpi',
                id: 'kpi-best-model',
                title: 'Best Model',
                apiEndpoint: '/api/astronomy/ml/models/performance',
                position: { row: 1, col: 3 },
                customRender: true
            },
            {
                type: 'line-chart',
                id: 'chart-regression-plot',
                title: 'Regression Plot (Predicted vs Actual)',
                apiEndpoint: '/api/astronomy/ml/models/regression',
                position: { row: 2, col: 1, span: 2 },
                isScatter: true
            },
            {
                type: 'bar-chart',
                id: 'chart-feature-importance-xgb',
                title: 'XGBoost Feature Importance',
                apiEndpoint: '/api/astronomy/ml/models/feature-importance',
                position: { row: 3, col: 1 },
                customRender: true
            },
            {
                type: 'bar-chart',
                id: 'chart-feature-importance-lgbm',
                title: 'LightGBM Feature Importance',
                apiEndpoint: '/api/astronomy/ml/models/feature-importance',
                position: { row: 3, col: 2 },
                customRender: true
            },
            {
                type: 'line-chart',
                id: 'chart-residuals',
                title: 'Residuals Plot',
                apiEndpoint: '/api/astronomy/ml/models/regression',
                position: { row: 4, col: 1, span: 2 },
                customRender: true
            }
        ]
    }
};

// TODO (USER): Implement function to load dashboard configuration
// TODO (USER): Implement function to initialize dashboard components
// TODO (USER): Add dashboard data fetching and caching

/**
 * Download Sample Astronomy Data
 * Adds UI button and handler for downloading real astronomy datasets
 */
function setupAstronomyDownloadButton() {
    // TODO (USER): Add download button to dashboard UI
    // Example: Add button to sidebar or dashboard header
    // const downloadBtn = document.createElement('button');
    // downloadBtn.textContent = '📥 Download Sample Astronomy Data';
    // downloadBtn.className = 'download-btn';
    // downloadBtn.addEventListener('click', handleAstronomyDownload);
    // document.getElementById('dashboard-sidebar').appendChild(downloadBtn);
}

async function handleAstronomyDownload(sourceName = null) {
    /**
     * Handle astronomy data download request.
     * 
     * TODO (USER): Implement download handler:
     * 1. Show loading indicator
     * 2. Call POST /api/astronomy/data/download with source_name
     * 3. Display download progress/status
     * 4. On success, refresh dashboard data
     * 5. Show validation results
     */
    
    // TODO (USER): Implement actual download call
    // try {
    //     const response = await fetch('/api/astronomy/data/download', {
    //         method: 'POST',
    //         headers: {'Content-Type': 'application/json'},
    //         body: JSON.stringify({source_name: sourceName})
    //     });
    //     const result = await response.json();
    //     
    //     if (result.status === 'success') {
    //         // Refresh dashboard
    //         if (window.layoutManager) {
    //             window.layoutManager.loadDashboard('astronomy', 'overview');
    //         }
    //         alert(`Download complete: ${result.file_path}`);
    //     } else {
    //         alert(`Download failed: ${result.error || result.message}`);
    //     }
    // } catch (error) {
    //     console.error('Download error:', error);
    //     alert('Download failed: ' + error.message);
    // }
    
    console.log('TODO: Implement astronomy data download');
    alert('Download functionality not yet implemented. Complete download_astronomy_sample() in data_sources/astronomy_download.py');
}

// Initialize download button when astronomy dashboards load
// TODO (USER): Call setupAstronomyDownloadButton() when astronomy mode is activated
