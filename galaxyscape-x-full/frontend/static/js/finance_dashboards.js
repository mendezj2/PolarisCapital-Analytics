/**
 * Finance Dashboards Configuration
 * Defines dashboard layouts and component configurations for Finance domain
 * TODO (USER): Implement dashboard-specific data loading and component initialization
 */

const FinanceDashboards = {
    /**
     * Risk Dashboard Configuration
     */
    risk: {
        layout: 'grid',
        requiredFeatures: ['returns', 'price', 'date'],
        globalFilters: [
            {
                name: 'date_start',
                label: 'Start Date',
                type: 'date-range',
                defaultStart: '2023-01-01',
                defaultEnd: '2024-01-01'
            },
            {
                name: 'tickers',
                label: 'Search Ticker',
                type: 'search',
                placeholder: 'Search by ticker symbol...',
                debounce: 300
            },
            {
                name: 'sectors',
                label: 'Sectors',
                type: 'multi-select',
                options: [
                    { value: 'Technology', label: 'Technology' },
                    { value: 'Finance', label: 'Finance' },
                    { value: 'Retail', label: 'Retail' },
                    { value: 'Energy', label: 'Energy' }
                ],
                defaultValue: []
            },
            {
                name: 'risk_min',
                label: 'Min Risk Score',
                type: 'slider',
                min: 0,
                max: 100,
                step: 1,
                defaultValue: 0
            },
            {
                name: 'risk_max',
                label: 'Max Risk Score',
                type: 'slider',
                min: 0,
                max: 100,
                step: 1,
                defaultValue: 100
            },
            {
                name: 'volatility_min',
                label: 'Min Volatility',
                type: 'number',
                defaultMin: 0,
                defaultMax: 1
            },
            {
                name: 'volatility_max',
                label: 'Max Volatility',
                type: 'number',
                defaultMin: 0,
                defaultMax: 1
            },
            {
                name: 'use_ml_predictions',
                label: 'Use ML Predictions',
                type: 'toggle',
                onLabel: 'ML On',
                offLabel: 'ML Off',
                defaultValue: false
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-portfolio-risk',
                title: 'Portfolio Risk Score',
                apiEndpoint: '/api/finance/risk_kpis',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-var',
                title: 'VaR (95%)',
                apiEndpoint: '/api/finance/risk_kpis',
                position: { row: 1, col: 2 },
                useFilteredData: true
            },
            {
                type: 'gauge',
                id: 'gauge-risk-level',
                title: 'Risk Level',
                apiEndpoint: '/api/finance/risk_kpis',
                position: { row: 1, col: 3 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-risk-trends',
                title: 'Risk Trends Over Time',
                apiEndpoint: '/api/finance/risk_timeseries',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-risk-breakdown',
                title: 'Risk by Asset',
                apiEndpoint: '/api/finance/dashboard/trends?metric=risk_breakdown',
                position: { row: 3, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-risky-assets',
                title: 'Top Risky Assets',
                apiEndpoint: '/api/finance/dashboard/leaderboard?metric=risk',
                position: { row: 4, col: 1 },
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-risk-details',
                title: 'Risk Details',
                apiEndpoint: '/api/finance/dashboard/trends?metric=risk_table',
                position: { row: 4, col: 2 },
                useFilteredData: true
            }
        ]
    },

    /**
     * Streaming Dashboard Configuration
     */
    streaming: {
        layout: 'grid',
        requiredFeatures: ['returns', 'price', 'date', 'ticker'],
        globalFilters: [
            {
                name: 'tickers',
                label: 'Search Ticker',
                type: 'search',
                placeholder: 'Filter by ticker...',
                debounce: 300
            },
            {
                name: 'sectors',
                label: 'Sectors',
                type: 'checkbox',
                options: [
                    { value: 'Technology', label: 'Technology' },
                    { value: 'Finance', label: 'Finance' },
                    { value: 'Retail', label: 'Retail' },
                    { value: 'Energy', label: 'Energy' }
                ],
                defaultValue: []
            },
            {
                name: 'correlation_threshold',
                label: 'Correlation Threshold',
                type: 'slider',
                min: 0,
                max: 1,
                step: 0.05,
                defaultValue: 0.5
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
                type: 'kpi',
                id: 'kpi-streaming-status',
                title: 'Streaming Status',
                apiEndpoint: '/api/finance/stream/latest',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'streaming-chart',
                id: 'chart-volatility-stream',
                title: 'Real-Time Volatility',
                apiEndpoint: '/api/finance/stream/risk',
                position: { row: 1, col: 2, span: 2 },
                updateInterval: 3000,
                useFilteredData: true
            },
            {
                type: 'gauge',
                id: 'gauge-current-risk',
                title: 'Current Risk',
                apiEndpoint: '/api/finance/stream/latest',
                position: { row: 2, col: 1 },
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-streaming-data',
                title: 'Latest Risk Scores',
                apiEndpoint: '/api/finance/stream/risk',
                position: { row: 2, col: 2, span: 2 },
                useFilteredData: true
            },
            {
                type: 'network-graph',
                id: 'network-streaming',
                title: 'Live Correlation Network',
                apiEndpoint: '/api/finance/stream/graph',
                position: { row: 3, col: 1, span: 2, fullWidth: true },
                useFilteredData: true
            }
        ]
    },

    /**
     * Correlation Network Dashboard Configuration
     */
    correlation: {
        layout: 'grid',
        requiredFeatures: ['returns', 'price'],
        globalFilters: [
            {
                name: 'correlation_threshold',
                label: 'Correlation Threshold',
                type: 'slider',
                min: 0,
                max: 1,
                step: 0.05,
                defaultValue: 0.5
            },
            {
                name: 'tickers',
                label: 'Search Ticker',
                type: 'search',
                placeholder: 'Search by ticker...',
                debounce: 300
            },
            {
                name: 'sectors',
                label: 'Sectors',
                type: 'multi-select',
                options: [
                    { value: 'Technology', label: 'Technology' },
                    { value: 'Finance', label: 'Finance' },
                    { value: 'Retail', label: 'Retail' },
                    { value: 'Energy', label: 'Energy' }
                ],
                defaultValue: []
            },
            {
                name: 'risk_min',
                label: 'Min Risk Score',
                type: 'slider',
                min: 0,
                max: 100,
                step: 1,
                defaultValue: 0
            },
            {
                name: 'risk_max',
                label: 'Max Risk Score',
                type: 'slider',
                min: 0,
                max: 100,
                step: 1,
                defaultValue: 100
            }
        ],
        components: [
            {
                type: 'network-graph',
                id: 'network-correlation',
                title: 'Asset Correlation Network',
                apiEndpoint: '/api/finance/correlation_network?threshold=0.5',
                position: { row: 1, col: 1, span: 2, fullWidth: true },
                useFilteredData: true
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-correlated',
                title: 'Most Correlated Pairs',
                apiEndpoint: '/api/finance/dashboard/leaderboard?metric=correlation',
                position: { row: 2, col: 1 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-correlation-matrix',
                title: 'Correlation Heatmap',
                apiEndpoint: '/api/finance/dashboard/trends?metric=correlation_matrix',
                position: { row: 2, col: 2 },
                useFilteredData: true
            }
        ]
    },

    /**
     * Portfolio Dashboard Configuration
     */
    portfolio: {
        layout: 'grid',
        requiredFeatures: ['price', 'ticker', 'sector'],
        globalFilters: [
            {
                name: 'date_range',
                label: 'Date Range',
                type: 'date-range',
                defaultStart: '2023-01-01',
                defaultEnd: '2024-01-01'
            },
            {
                name: 'tickers',
                label: 'Search Ticker',
                type: 'search',
                placeholder: 'Search holdings...',
                debounce: 300
            },
            {
                name: 'sectors',
                label: 'Sectors',
                type: 'multi-select',
                options: [
                    { value: 'Technology', label: 'Technology' },
                    { value: 'Finance', label: 'Finance' },
                    { value: 'Retail', label: 'Retail' },
                    { value: 'Energy', label: 'Energy' }
                ],
                defaultValue: []
            },
            {
                name: 'expected_return',
                label: 'Expected Return (%)',
                type: 'slider',
                min: -10,
                max: 20,
                step: 0.5,
                defaultValue: 5
            },
            {
                name: 'time_horizon',
                label: 'Time Horizon (years)',
                type: 'slider',
                min: 1,
                max: 10,
                step: 1,
                defaultValue: 5
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-portfolio-value',
                title: 'Portfolio Value',
                apiEndpoint: '/api/finance/dashboard/kpi?metric=portfolio_value',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-sharpe',
                title: 'Sharpe Ratio',
                apiEndpoint: '/api/finance/dashboard/kpi?metric=sharpe',
                position: { row: 1, col: 2 },
                useFilteredData: true
            },
            {
                type: 'pie-chart',
                id: 'chart-allocation',
                title: 'Portfolio Allocation',
                apiEndpoint: '/api/finance/dashboard/trends?metric=allocation',
                position: { row: 1, col: 3 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-returns',
                title: 'Portfolio Returns',
                apiEndpoint: '/api/finance/dashboard/trends?metric=returns',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'map-card',
                id: 'map-geographic',
                title: 'Geographic Risk Map',
                apiEndpoint: '/api/finance/dashboard/map',
                position: { row: 3, col: 1 },
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-holdings',
                title: 'Holdings',
                apiEndpoint: '/api/finance/dashboard/trends?metric=holdings',
                position: { row: 3, col: 2 },
                useFilteredData: true
            }
        ]
    },

    /**
     * Compliance & Audit Dashboard Configuration
     */
    compliance: {
        layout: 'grid',
        requiredFeatures: [],
        globalFilters: [
            {
                name: 'status',
                label: 'Status',
                type: 'dropdown',
                options: [
                    { value: 'all', label: 'All' },
                    { value: 'compliant', label: 'Compliant' },
                    { value: 'non-compliant', label: 'Non-Compliant' }
                ],
                defaultValue: 'all'
            },
            {
                name: 'risk_level',
                label: 'Risk Level',
                type: 'dropdown',
                options: [
                    { value: 'all', label: 'All' },
                    { value: 'Low', label: 'Low' },
                    { value: 'Medium', label: 'Medium' },
                    { value: 'High', label: 'High' },
                    { value: 'Critical', label: 'Critical' }
                ],
                defaultValue: 'all'
            },
            {
                name: 'date_range',
                label: 'Date Range',
                type: 'date-range',
                defaultStart: '2023-01-01',
                defaultEnd: '2024-01-01'
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-compliant',
                title: 'Compliant Items',
                apiEndpoint: '/api/finance/compliance_summary',
                position: { row: 1, col: 1 },
                customRender: true,
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-non-compliant',
                title: 'Non-Compliant Items',
                apiEndpoint: '/api/finance/compliance_summary',
                position: { row: 1, col: 2 },
                customRender: true,
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-compliance-rate',
                title: 'Compliance Rate',
                apiEndpoint: '/api/finance/compliance_summary',
                position: { row: 1, col: 3 },
                customRender: true,
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-audit-log',
                title: 'Audit Log',
                apiEndpoint: '/api/finance/compliance_summary',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            }
        ]
    },
    
    /**
     * Stock Explorer Dashboard Configuration
     * Real-time stock and index fund exploration and comparison
     */
    'stock-explorer': {
        layout: 'grid',
        requiredFeatures: ['price', 'ticker', 'sector'],
        globalFilters: [
            {
                name: 'sector',
                label: 'Sector',
                type: 'dropdown',
                options: [
                    { value: 'all', label: 'All Sectors' },
                    { value: 'Technology', label: 'Technology' },
                    { value: 'Financial Services', label: 'Financial Services' },
                    { value: 'Healthcare', label: 'Healthcare' },
                    { value: 'Consumer Cyclical', label: 'Consumer Cyclical' },
                    { value: 'Energy', label: 'Energy' }
                ],
                defaultValue: 'all'
            },
            {
                name: 'sort_by',
                label: 'Sort By',
                type: 'dropdown',
                options: [
                    { value: 'market_cap', label: 'Market Cap' },
                    { value: 'change_percent', label: 'Change %' },
                    { value: 'volume', label: 'Volume' },
                    { value: 'pe_ratio', label: 'P/E Ratio' }
                ],
                defaultValue: 'market_cap'
            }
        ],
        components: [
            {
                type: 'stock-symbol-input',
                id: 'stock-symbol-input',
                title: 'Add Stock/Index Symbols',
                apiEndpoint: null,
                position: { row: 1, col: 1, span: 2 },
                useFilteredData: false
            },
            {
                type: 'analyze-button',
                id: 'analyze-stocks-btn',
                title: 'Analyze Stocks',
                apiEndpoint: '/api/finance/stock/analyze',
                position: { row: 1, col: 3 },
                useFilteredData: false
            },
            {
                type: 'kpi',
                id: 'kpi-total-stocks',
                title: 'Stocks Tracked',
                apiEndpoint: '/api/finance/stock/explore',
                position: { row: 2, col: 1 },
                useFilteredData: false
            },
            {
                type: 'data-table',
                id: 'table-stock-explorer',
                title: 'Stock Comparison Table',
                apiEndpoint: '/api/finance/stock/explore',
                position: { row: 3, col: 1, span: 3, fullWidth: true },
                useFilteredData: false
            },
            {
                type: 'bar-chart',
                id: 'chart-price-comparison',
                title: 'Price Comparison',
                apiEndpoint: '/api/finance/stock/compare',
                position: { row: 4, col: 1 },
                useFilteredData: false
            },
            {
                type: 'line-chart',
                id: 'chart-returns-comparison',
                title: 'Returns Comparison (1M, 3M, 1Y)',
                apiEndpoint: '/api/finance/stock/compare',
                position: { row: 4, col: 2 },
                useFilteredData: false
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-top-performers',
                title: 'Top Performers',
                apiEndpoint: '/api/finance/stock/explore',
                position: { row: 2, col: 2 },
                useFilteredData: false
            },
            {
                type: 'pie-chart',
                id: 'chart-sector-breakdown',
                title: 'Sector Breakdown',
                apiEndpoint: '/api/finance/stock/explore',
                position: { row: 2, col: 3 },
                useFilteredData: false
            }
        ]
    },
    
    /**
     * Future Outcomes Assessment Dashboard Configuration
     * Project future portfolio outcomes based on current holdings
     */
    'future-outcomes': {
        layout: 'grid',
        requiredFeatures: ['price', 'returns'],
        globalFilters: [
            {
                name: 'time_horizon',
                label: 'Time Horizon (years)',
                type: 'slider',
                min: 0.5,
                max: 10,
                step: 0.5,
                defaultValue: 1
            },
            {
                name: 'confidence_level',
                label: 'Confidence Level',
                type: 'dropdown',
                options: [
                    { value: 0.90, label: '90%' },
                    { value: 0.95, label: '95%' },
                    { value: 0.99, label: '99%' }
                ],
                defaultValue: 0.95
            }
        ],
        components: [
            {
                type: 'portfolio-input',
                id: 'portfolio-input',
                title: 'Portfolio Holdings',
                apiEndpoint: null,
                position: { row: 1, col: 1, span: 2 },
                useFilteredData: false
            },
            {
                type: 'kpi',
                id: 'kpi-current-value',
                title: 'Current Portfolio Value',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 2, col: 1 },
                useFilteredData: false
            },
            {
                type: 'kpi',
                id: 'kpi-projected-value',
                title: 'Projected Value (Mean)',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 2, col: 2 },
                useFilteredData: false
            },
            {
                type: 'kpi',
                id: 'kpi-expected-return',
                title: 'Expected Return %',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 2, col: 3 },
                useFilteredData: false
            },
            {
                type: 'gauge',
                id: 'gauge-volatility',
                title: 'Portfolio Volatility',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 3, col: 1 },
                useFilteredData: false
            },
            {
                type: 'bar-chart',
                id: 'chart-scenarios',
                title: 'Future Scenarios',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 3, col: 2, span: 2 },
                useFilteredData: false
            },
            {
                type: 'line-chart',
                id: 'chart-projection-distribution',
                title: 'Projection Distribution',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 4, col: 1, span: 2 },
                useFilteredData: false
            },
            {
                type: 'data-table',
                id: 'table-scenarios',
                title: 'Detailed Scenarios',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 5, col: 1, span: 2 },
                useFilteredData: false
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-recommendations',
                title: 'Recommendations',
                apiEndpoint: '/api/finance/future/outcomes',
                position: { row: 6, col: 1, span: 2 },
                useFilteredData: false
            }
        ]
    },

    /**
     * Marketing Analytics / Signage Evaluation Dashboard
     */
    'marketing-analytics': {
        layout: 'grid',
        requiredFeatures: ['roi', 'price'],
        globalFilters: [
            {
                name: 'time_period',
                label: 'Time Period',
                type: 'single-select',
                options: [
                    { value: 'last_month', label: 'Last Month' },
                    { value: 'last_quarter', label: 'Last Quarter' },
                    { value: 'last_year', label: 'Last Year' },
                    { value: 'ytd', label: 'Year to Date' }
                ],
                defaultValue: 'last_quarter'
            },
            {
                name: 'channel',
                label: 'Channel',
                type: 'multi-select',
                options: [
                    { value: 'in_store', label: 'In-Store' },
                    { value: 'online', label: '.com Website' },
                    { value: 'mobile', label: 'Mobile App' },
                    { value: 'social', label: 'Social Media' }
                ],
                defaultValue: []
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-total-roi',
                title: 'Total ROI',
                apiEndpoint: '/api/finance/marketing/signage-evaluation',
                position: { row: 1, col: 1 },
                customRender: true
            },
            {
                type: 'kpi',
                id: 'kpi-sync-score',
                title: 'Omni-Channel Sync Score',
                apiEndpoint: '/api/finance/marketing/omni-channel',
                position: { row: 1, col: 2 },
                customRender: true
            },
            {
                type: 'kpi',
                id: 'kpi-model-accuracy',
                title: 'Predictability Model Accuracy',
                apiEndpoint: '/api/finance/marketing/predictability-model',
                position: { row: 1, col: 3 },
                customRender: true
            },
            {
                type: 'bar-chart',
                id: 'chart-signage-roi',
                title: 'Signage ROI by Location',
                apiEndpoint: '/api/finance/marketing/signage-evaluation',
                position: { row: 2, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'line-chart',
                id: 'chart-omni-channel',
                title: 'Omni-Channel Performance',
                apiEndpoint: '/api/finance/marketing/omni-channel',
                position: { row: 3, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'bar-chart',
                id: 'chart-predicted-initiatives',
                title: 'Predicted Marketing Initiatives ROI',
                apiEndpoint: '/api/finance/marketing/predictability-model',
                position: { row: 4, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'pie-chart',
                id: 'chart-budget-allocation',
                title: 'Current Budget Allocation',
                apiEndpoint: '/api/finance/marketing/expenditure-optimization',
                position: { row: 5, col: 1 },
                customRender: true
            },
            {
                type: 'pie-chart',
                id: 'chart-optimized-allocation',
                title: 'Optimized Budget Allocation',
                apiEndpoint: '/api/finance/marketing/expenditure-optimization',
                position: { row: 5, col: 2 },
                customRender: true
            },
            {
                type: 'line-chart',
                id: 'chart-pro-journey',
                title: 'PRO Consumer Journey Funnel',
                apiEndpoint: '/api/finance/marketing/pro-consumer-journey',
                position: { row: 6, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'data-table',
                id: 'table-signage-assets',
                title: 'In-Store Assets ROI Analysis',
                apiEndpoint: '/api/finance/marketing/signage-evaluation',
                position: { row: 7, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'data-table',
                id: 'table-channel-expansion',
                title: 'Channel Expansion Strategy (Lowe\'s & Home Depot)',
                apiEndpoint: '/api/finance/marketing/pro-consumer-journey',
                position: { row: 8, col: 1, span: 2 },
                customRender: true
            },
            {
                type: 'leaderboard',
                id: 'leaderboard-recommendations',
                title: 'Marketing Recommendations',
                apiEndpoint: '/api/finance/marketing/signage-evaluation',
                position: { row: 9, col: 1, span: 2 },
                customRender: true
            }
        ]
    },

    /**
     * ML Models & Regression Dashboard Configuration
     */
    'ml-models': {
        layout: 'grid',
        requiredFeatures: ['returns', 'price'],
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
            },
            {
                name: 'feature_count',
                label: 'Top Features',
                type: 'slider',
                min: 5,
                max: 20,
                step: 1,
                defaultValue: 10
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-xgb-mae',
                title: 'XGBoost MAE',
                apiEndpoint: '/api/finance/ml/models/performance?model=xgboost&metric=mae',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-lgbm-mae',
                title: 'LightGBM MAE',
                apiEndpoint: '/api/finance/ml/models/performance?model=lightgbm&metric=mae',
                position: { row: 1, col: 2 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-best-model',
                title: 'Best Model (MAE)',
                apiEndpoint: '/api/finance/ml/models/performance?metric=best_model',
                position: { row: 1, col: 3 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-regression-plot',
                title: 'Predicted vs Actual Risk Score',
                apiEndpoint: '/api/finance/ml/models/regression',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-residuals-plot',
                title: 'Residuals Plot',
                apiEndpoint: '/api/finance/ml/models/residuals',
                position: { row: 3, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-feature-importance',
                title: 'Feature Importance',
                apiEndpoint: '/api/finance/ml/models/feature-importance',
                position: { row: 4, col: 1, span: 2 },
                useFilteredData: true
            }
        ]
    },

    /**
     * Game Theory Analysis Dashboard Configuration
     */
    'game-theory': {
        layout: 'grid',
        requiredFeatures: ['returns', 'price'],
        globalFilters: [
            {
                name: 'game_type',
                label: 'Game Theory Model',
                type: 'single-select',
                options: [
                    { value: 'nash', label: 'Nash Equilibrium' },
                    { value: 'shapley', label: 'Shapley Value' },
                    { value: 'prisoner', label: 'Prisoner\'s Dilemma' },
                    { value: 'auction', label: 'Auction Theory' },
                    { value: 'evolutionary', label: 'Evolutionary Dynamics' }
                ],
                defaultValue: 'nash'
            },
            {
                name: 'risk_free_rate',
                label: 'Risk-Free Rate',
                type: 'slider',
                min: 0,
                max: 0.1,
                step: 0.001,
                defaultValue: 0.02
            }
        ],
        components: [
            {
                type: 'kpi',
                id: 'kpi-nash-sharpe',
                title: 'Nash Equilibrium Sharpe',
                apiEndpoint: '/api/finance/game-theory/nash-equilibrium',
                position: { row: 1, col: 1 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-nash-return',
                title: 'Expected Return',
                apiEndpoint: '/api/finance/game-theory/nash-equilibrium',
                position: { row: 1, col: 2 },
                useFilteredData: true
            },
            {
                type: 'kpi',
                id: 'kpi-nash-volatility',
                title: 'Portfolio Volatility',
                apiEndpoint: '/api/finance/game-theory/nash-equilibrium',
                position: { row: 1, col: 3 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-nash-weights',
                title: 'Nash Equilibrium Weights',
                apiEndpoint: '/api/finance/game-theory/nash-equilibrium',
                position: { row: 2, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-shapley-values',
                title: 'Shapley Values (Portfolio Contribution)',
                apiEndpoint: '/api/finance/game-theory/shapley-value',
                position: { row: 3, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'data-table',
                id: 'table-prisoner-dilemma',
                title: 'Prisoner\'s Dilemma Payoff Matrix',
                apiEndpoint: '/api/finance/game-theory/prisoner-dilemma',
                position: { row: 4, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'line-chart',
                id: 'chart-evolutionary-dynamics',
                title: 'Evolutionary Strategy Dynamics',
                apiEndpoint: '/api/finance/game-theory/evolutionary',
                position: { row: 5, col: 1, span: 2 },
                useFilteredData: true
            },
            {
                type: 'bar-chart',
                id: 'chart-auction-bids',
                title: 'Auction Theory - Optimal Bids',
                apiEndpoint: '/api/finance/game-theory/auction',
                position: { row: 6, col: 1, span: 2 },
                useFilteredData: true
            }
        ]
    }
};

// TODO (USER): Implement function to load dashboard configuration
// TODO (USER): Implement function to initialize dashboard components
// TODO (USER): Add streaming data polling for real-time updates

/**
 * Download Sample Finance Data
 * Adds UI button and handler for downloading real finance datasets
 */
function setupFinanceDownloadButton() {
    // TODO (USER): Add download button to dashboard UI
    // Example: Add button to sidebar or dashboard header
    // const downloadBtn = document.createElement('button');
    // downloadBtn.textContent = '📥 Download Sample Finance Data';
    // downloadBtn.className = 'download-btn';
    // downloadBtn.addEventListener('click', handleFinanceDownload);
    // document.getElementById('dashboard-sidebar').appendChild(downloadBtn);
}

async function handleFinanceDownload(sourceName = null, tickers = ['AAPL', 'MSFT', 'GOOGL']) {
    /**
     * Handle finance data download request.
     * 
     * TODO (USER): Implement download handler:
     * 1. Show loading indicator
     * 2. Call POST /api/finance/data/download with source_name and tickers
     * 3. Display download progress/status
     * 4. On success, refresh dashboard data
     * 5. Show validation results
     */
    
    // TODO (USER): Implement actual download call
    // try {
    //     const response = await fetch('/api/finance/data/download', {
    //         method: 'POST',
    //         headers: {'Content-Type': 'application/json'},
    //         body: JSON.stringify({
    //             source_name: sourceName,
    //             tickers: tickers
    //         })
    //     });
    //     const result = await response.json();
    //     
    //     if (result.status === 'success') {
    //         // Refresh dashboard
    //         if (window.layoutManager) {
    //             window.layoutManager.loadDashboard('finance', 'risk');
    //         }
    //         alert(`Download complete: ${result.file_path}`);
    //     } else {
    //         alert(`Download failed: ${result.error || result.message}`);
    //     }
    // } catch (error) {
    //     console.error('Download error:', error);
    //     alert('Download failed: ' + error.message);
    // }
    
    console.log('TODO: Implement finance data download');
    alert('Download functionality not yet implemented. Complete download_finance_sample() in data_sources/finance_download.py');
}

// Initialize download button when finance dashboards load
// TODO (USER): Call setupFinanceDownloadButton() when finance mode is activated
