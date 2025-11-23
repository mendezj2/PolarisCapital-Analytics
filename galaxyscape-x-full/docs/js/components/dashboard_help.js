/**
 * Dashboard Help Modal Component
 * Shows descriptions and explanations for each dashboard
 */

class DashboardHelp {
    constructor() {
        this.modal = null;
        this.descriptions = this.loadDescriptions();
    }

    /**
     * Load dashboard descriptions
     */
    loadDescriptions() {
        return {
            // Astronomy Dashboards
            'astronomy': {
                'overview': {
                    title: 'Astronomy Overview Dashboard',
                    description: 'This dashboard provides a comprehensive overview of stellar data analysis. It displays key metrics including total stars analyzed, average stellar age, age distribution trends, and cluster information.',
                    significance: 'The Overview Dashboard serves as the central command center for stellar analysis, providing immediate insights into dataset characteristics, population demographics, and structural patterns. It enables rapid assessment of data quality, sample representativeness, and overall stellar population properties.',
                    purpose: 'Primary purpose: (1) Quick assessment of dataset scale and quality, (2) Identification of population characteristics (age distribution, cluster structure), (3) Foundation for deeper analysis by establishing baseline metrics, (4) Quality control by detecting anomalies or biases in the dataset.',
                    metrics: [
                        'Total Stars: Number of stellar objects in the dataset',
                        'Average Age: Mean estimated age of stars in billions of years',
                        'Age Distribution: Temporal trends showing how stellar ages are distributed',
                        'Cluster Sizes: Number of stars in each identified cluster',
                        'Cluster Distribution: Visual breakdown of cluster membership'
                    ],
                    analysis: 'Use this dashboard to get a high-level understanding of your stellar dataset. The metrics help identify patterns in stellar populations and cluster formations.'
                },
                'star-explorer': {
                    title: 'Star Explorer Dashboard',
                    description: 'Interactive exploration tool for analyzing individual stars and their properties. Filter and search through stellar data to find specific stars based on rotation period, color index, mass, and cluster membership.',
                    significance: 'The Star Explorer Dashboard enables detailed investigation of individual stellar objects, facilitating hypothesis-driven research and discovery of unusual stars. It supports data-driven astronomy by allowing researchers to identify stars matching specific criteria, discover outliers, and validate theoretical predictions through empirical observation.',
                    purpose: 'Primary purpose: (1) Individual star investigation and detailed analysis, (2) Hypothesis testing by filtering for specific stellar properties, (3) Outlier detection and rare star discovery, (4) Data quality assessment through detailed inspection, (5) Comparative analysis of stellar properties across the dataset.',
                    metrics: [
                        'Star Table: Detailed data table with all stellar properties',
                        'Scatter Plot: Visualize relationships between stellar parameters (e.g., color index vs rotation period)',
                        'Filters: Use filters to narrow down stars by specific criteria'
                    ],
                    analysis: 'This dashboard helps you explore and understand individual stellar objects. Use filters to identify stars with specific characteristics or find outliers in the dataset.'
                },
                'sky-map': {
                    title: 'Sky Map / Projection Dashboard',
                    description: 'Visual representation of stars in celestial coordinates. Shows the spatial distribution of stars using Right Ascension (RA) and Declination (Dec) coordinates, similar to a star chart.',
                    metrics: [
                        'RA (Right Ascension): Celestial longitude, measured in hours (0-24h)',
                        'Dec (Declination): Celestial latitude, measured in degrees (-90° to +90°)',
                        'Brightness: Visual magnitude or flux values',
                        'Zoom/Pan: Interactive navigation of the sky map'
                    ],
                    analysis: 'Use this dashboard to visualize the spatial distribution of stars in the sky. It helps identify star clusters, constellations, and spatial patterns in the stellar data.'
                },
                'light-curve': {
                    title: 'Light Curve / Time Series Dashboard',
                    description: 'Analyzes temporal variations in stellar brightness. Displays light curves showing how a star\'s brightness changes over time, which is crucial for detecting exoplanets, variable stars, and stellar activity.',
                    metrics: [
                        'Time Series: Flux or magnitude measurements over time',
                        'Star Selection: Choose specific stars to analyze',
                        'Variability: Identify periodic or irregular brightness changes'
                    ],
                    analysis: 'Light curves reveal important information about stellar behavior. Periodic variations may indicate exoplanet transits, stellar rotation, or binary systems. Irregular variations can signal stellar flares or other transient events.'
                },
                'clusters': {
                    title: 'Cluster Analysis Dashboard',
                    description: 'Identifies and visualizes stellar clusters using machine learning clustering algorithms (K-means, DBSCAN, HDBSCAN). Groups stars with similar properties together.',
                    significance: 'The Cluster Analysis Dashboard reveals hidden structure and patterns in stellar populations through unsupervised machine learning. It enables discovery of stellar associations, common evolutionary stages, and physical relationships that may not be apparent through visual inspection. Critical for understanding stellar formation history and population demographics.',
                    purpose: 'Primary purpose: (1) Discovery of stellar associations and groups, (2) Identification of common evolutionary stages, (3) Pattern recognition in high-dimensional stellar data, (4) Data-driven classification of stellar populations, (5) Validation of theoretical stellar evolution models through empirical clustering.',
                    metrics: [
                        'Cluster Count: Number of identified clusters',
                        'Cluster Members: Stars belonging to each cluster',
                        'Cluster Properties: Average characteristics of each cluster',
                        'Visualization: 2D/3D projection of clusters'
                    ],
                    analysis: 'Clustering helps identify groups of stars with similar properties, which may indicate common origins, evolutionary stages, or physical relationships. Use this to discover patterns in stellar populations.'
                },
                'anomalies': {
                    title: 'Anomaly Detection Dashboard',
                    description: 'Identifies unusual or outlier stars using anomaly detection algorithms (Isolation Forest, LOF). Highlights stars with properties that deviate significantly from the norm.',
                    metrics: [
                        'Anomaly Score: Numerical score indicating how unusual a star is',
                        'Anomaly Count: Number of detected outliers',
                        'Anomaly Properties: Characteristics of unusual stars',
                        'Visualization: Highlighted anomalies in data visualizations'
                    ],
                    analysis: 'Anomalies may represent rare stellar types, data quality issues, or interesting astrophysical phenomena. Investigate high-scoring anomalies to discover unique stars or potential data errors.'
                },
                'shap-explainability': {
                    title: 'SHAP Explainability Dashboard',
                    description: 'Explains machine learning model predictions using SHAP (SHapley Additive exPlanations) values. Shows which features contribute most to model predictions for stellar age or other properties.',
                    metrics: [
                        'Feature Importance: Which stellar properties most influence predictions',
                        'SHAP Values: Contribution of each feature to individual predictions',
                        'Model Explanation: Understanding of how the ML model makes decisions'
                    ],
                    analysis: 'SHAP values help you understand why the model makes specific predictions. Features with high SHAP values have the most influence on the model\'s output, helping you interpret and trust the results.'
                }
            },
            // Finance Dashboards
            'finance': {
                'risk-overview': {
                    title: 'Risk Overview Dashboard',
                    description: 'Comprehensive risk assessment dashboard showing portfolio risk metrics, volatility trends, and key risk indicators. Helps monitor overall portfolio health and risk exposure.',
                    significance: 'The Risk Overview Dashboard is the primary risk management tool, providing real-time visibility into portfolio risk exposure. It enables proactive risk management by identifying risk trends before they become critical, supports regulatory compliance through documented risk metrics, and facilitates informed decision-making for portfolio adjustments.',
                    purpose: 'Primary purpose: (1) Real-time risk monitoring and alerting, (2) Regulatory compliance and risk reporting, (3) Portfolio optimization through risk-return analysis, (4) Early warning system for risk escalation, (5) Performance attribution by linking risk to returns.',
                    metrics: [
                        'Portfolio Risk Score: Overall risk level (0-100 scale)',
                        'VaR (Value at Risk): Maximum expected loss at 95% confidence',
                        'Volatility: Standard deviation of returns, indicating price variability',
                        'Risk Trends: Historical risk levels over time',
                        'Risk Level Gauge: Visual indicator of current risk status'
                    ],
                    analysis: 'Use this dashboard to monitor portfolio risk in real-time. Higher risk scores indicate greater potential for losses. Track trends to identify increasing risk exposure and take appropriate actions.'
                },
                'correlation-network': {
                    title: 'Correlation Network Dashboard',
                    description: 'Visualizes relationships between assets using correlation analysis. Shows how assets move together, helping identify diversification opportunities and potential contagion risks.',
                    significance: 'The Correlation Network Dashboard reveals hidden relationships and systemic risks in financial markets. It enables portfolio optimization by identifying diversification opportunities, detects contagion risks during market stress, and supports risk management through understanding of asset interdependencies. Network topology reveals market structure and systemic importance of assets.',
                    purpose: 'Primary purpose: (1) Portfolio diversification optimization, (2) Contagion risk detection and systemic risk assessment, (3) Sector and cluster identification, (4) Hedging strategy development through negative correlations, (5) Market structure analysis and network topology understanding.',
                    metrics: [
                        'Correlation Strength: How closely assets move together (-1 to +1)',
                        'Network Nodes: Individual assets or securities',
                        'Network Edges: Connections showing correlation relationships',
                        'Sector Clusters: Groups of assets in the same sector',
                        'Correlation Threshold: Filter to show only strong correlations'
                    ],
                    analysis: 'High positive correlations indicate assets move together, reducing diversification benefits. Negative correlations can provide hedging opportunities. Use this to optimize portfolio diversification and identify systemic risks.'
                },
                'streaming-live-risk': {
                    title: 'Streaming / Live Risk Dashboard',
                    description: 'Real-time risk monitoring using streaming data from Kafka or live market feeds. Updates continuously to show current risk levels, volatility, and recent risk events.',
                    metrics: [
                        'Live Risk Score: Current risk level updated in real-time',
                        'Streaming Data: Latest risk calculations from live market data',
                        'Recent Events: Timeline of risk events and anomalies',
                        'Live Correlation Network: Real-time asset correlation updates'
                    ],
                    analysis: 'Monitor risk in real-time to respond quickly to market changes. Sudden spikes in risk scores may indicate market stress or portfolio issues requiring immediate attention.'
                },
                'compliance-audit': {
                    title: 'Compliance & Audit Dashboard',
                    description: 'Tracks regulatory compliance and audit requirements. Monitors compliance status, risk levels, and maintains an audit log of compliance-related events.',
                    metrics: [
                        'Compliance Status: Number of compliant vs non-compliant items',
                        'Risk Distribution: Breakdown of risk levels across portfolio',
                        'Audit Log: Timeline of compliance events and changes',
                        'Compliance Filters: Filter by date, status, and risk level'
                    ],
                    analysis: 'Ensure your portfolio meets regulatory requirements. Non-compliant items may require immediate action. Use audit logs to track compliance history and demonstrate regulatory adherence.'
                },
                'stock-explorer': {
                    title: 'Stock Explorer & Comparison Dashboard',
                    description: 'Explore and compare stocks, ETFs, and index funds in real-time. Analyze multiple securities side-by-side using current market data, financial metrics, and performance indicators.',
                    significance: 'The Stock Explorer Dashboard provides real-time market intelligence and comparative analysis capabilities. It enables informed investment decisions through side-by-side comparison, supports research workflows by aggregating multiple data sources, and facilitates stock selection through comprehensive metric analysis. The ML-based analysis feature provides deeper insights beyond basic metrics.',
                    purpose: 'Primary purpose: (1) Real-time stock research and comparison, (2) Investment opportunity identification and screening, (3) Performance benchmarking against peers, (4) ML-enhanced analysis for risk and return prediction, (5) Sector allocation and diversification analysis.',
                    metrics: [
                        'Stock Data: Current price, change, market cap, P/E ratio, dividend yield',
                        'Comparison Charts: Visual comparison of prices, returns, and metrics',
                        'Sector Breakdown: Distribution of selected stocks by sector',
                        'Top Performers: Leaderboard of best-performing stocks',
                        'Real-time Updates: Live data from Yahoo Finance'
                    ],
                    analysis: 'Use this dashboard to research and compare investment opportunities. Add multiple tickers to compare their performance, financial metrics, and sector exposure. The Analyze button performs deep ML-based analysis on selected stocks.'
                },
                'future-outcomes': {
                    title: 'Future Outcomes Assessment Dashboard',
                    description: 'Monte Carlo simulation for portfolio projection. Estimates future portfolio values based on historical returns, volatility, and user-defined assumptions. Provides confidence intervals for different time horizons.',
                    significance: 'The Future Outcomes Dashboard enables forward-looking portfolio planning and risk assessment through probabilistic modeling. It provides quantitative estimates of future portfolio performance under uncertainty, supports strategic planning by showing range of possible outcomes, and helps set realistic expectations through confidence intervals. Critical for long-term investment planning and risk budgeting.',
                    purpose: 'Primary purpose: (1) Portfolio projection and future value estimation, (2) Risk budgeting and scenario planning, (3) Goal-based investment planning (retirement, education, etc.), (4) Stress testing through worst-case scenarios, (5) Confidence interval estimation for decision-making under uncertainty.',
                    metrics: [
                        'Projected Value: Expected portfolio value at future dates',
                        'Confidence Intervals: Range of possible outcomes (e.g., 90%, 95%)',
                        'Expected Return: Mean projected return',
                        'Volatility Forecast: Expected price variability',
                        'Scenario Analysis: Best case, worst case, and median scenarios'
                    ],
                    analysis: 'Monte Carlo simulations help estimate future portfolio performance under uncertainty. Higher confidence levels show wider ranges of possible outcomes. Use this to plan for different market scenarios and set realistic expectations.'
                },
                'risk': {
                    title: 'Risk Dashboard',
                    description: 'Comprehensive risk analysis with multiple risk metrics, time series analysis, and risk distribution visualizations.',
                    metrics: [
                        'Risk KPIs: Key risk indicators and metrics',
                        'Risk Time Series: Historical risk trends',
                        'Risk Distribution: Breakdown of risk across portfolio'
                    ],
                    analysis: 'Monitor and analyze portfolio risk using multiple dimensions and time periods.'
                },
                'streaming': {
                    title: 'Streaming Analytics Dashboard',
                    description: 'Real-time streaming analytics for live market data and risk monitoring.',
                    metrics: [
                        'Live Updates: Real-time data streams',
                        'Streaming Metrics: Continuously updated risk and performance indicators'
                    ],
                    analysis: 'Track live market conditions and portfolio performance in real-time.'
                },
                'correlation': {
                    title: 'Correlation Network Dashboard',
                    description: 'Asset correlation analysis and network visualization.',
                    metrics: [
                        'Correlation Matrix: Pairwise correlations between assets',
                        'Network Graph: Visual representation of asset relationships'
                    ],
                    analysis: 'Understand how assets move together and identify diversification opportunities.'
                },
                'portfolio': {
                    title: 'Portfolio Dashboard',
                    description: 'Portfolio composition, allocation, and performance analysis.',
                    metrics: [
                        'Portfolio Allocation: Asset distribution',
                        'Performance Metrics: Returns, volatility, Sharpe ratio'
                    ],
                    analysis: 'Analyze portfolio composition and performance across different dimensions.'
                }
            }
        };
    }

    /**
     * Show help modal for current dashboard
     */
    showHelp(domain, dashboardName) {
        const desc = this.descriptions[domain]?.[dashboardName];
        if (!desc) {
            this.showGenericHelp(domain, dashboardName);
            return;
        }

        this.createModal(desc);
    }

    /**
     * Show generic help if specific description not found
     */
    showGenericHelp(domain, dashboardName) {
        const desc = {
            title: `${domain.charAt(0).toUpperCase() + domain.slice(1)} - ${dashboardName}`,
            description: `This dashboard provides analysis and visualization for ${domain} data. Use the filters and controls to explore the data and customize your view.`,
            metrics: [],
            analysis: 'Use the interactive filters and charts to explore your data. Click on chart elements for more details.'
        };
        this.createModal(desc);
    }

    /**
     * Create and show modal
     */
    createModal(desc) {
        // Remove existing modal
        const existing = document.getElementById('dashboard-help-modal');
        if (existing) {
            existing.remove();
        }

        // Create modal
        const modal = document.createElement('div');
        modal.id = 'dashboard-help-modal';
        modal.className = 'help-modal';
        modal.innerHTML = `
            <div class="help-modal-overlay"></div>
            <div class="help-modal-content">
                <div class="help-modal-header">
                    <h3>${desc.title}</h3>
                    <button class="help-modal-close" id="help-modal-close">
                        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828842.png" alt="Close" style="width: 20px; height: 20px;">
                    </button>
                </div>
                <div class="help-modal-body">
                    <div class="help-section">
                        <h4>Description</h4>
                        <p>${desc.description}</p>
                    </div>
                    ${desc.significance ? `
                    <div class="help-section">
                        <h4>Significance</h4>
                        <p>${desc.significance}</p>
                    </div>
                    ` : ''}
                    ${desc.purpose ? `
                    <div class="help-section">
                        <h4>Purpose</h4>
                        <p>${desc.purpose}</p>
                    </div>
                    ` : ''}
                    ${desc.metrics && desc.metrics.length > 0 ? `
                    <div class="help-section">
                        <h4>Key Metrics</h4>
                        <ul>
                            ${desc.metrics.map(m => `<li>${m}</li>`).join('')}
                        </ul>
                    </div>
                    ` : ''}
                    <div class="help-section">
                        <h4>How to Use</h4>
                        <p>${desc.analysis}</p>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        const closeBtn = document.getElementById('help-modal-close');
        const overlay = modal.querySelector('.help-modal-overlay');
        
        const closeModal = () => {
            modal.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => modal.remove(), 300);
        };

        if (closeBtn) {
            closeBtn.addEventListener('click', closeModal);
        }
        if (overlay) {
            overlay.addEventListener('click', closeModal);
        }

        // Show modal with animation
        setTimeout(() => {
            modal.style.opacity = '1';
        }, 10);
        
        // Close on Escape key
        const escapeHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escapeHandler);
            }
        };
        document.addEventListener('keydown', escapeHandler);
    }
}

// Global instance
window.dashboardHelp = new DashboardHelp();

