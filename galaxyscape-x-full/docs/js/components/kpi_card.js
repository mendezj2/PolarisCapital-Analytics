/**
 * KPI Card Component
 * Displays key performance indicators with values and trends
 * TODO (USER): Implement real data fetching and trend calculations
 */

class KPICard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'KPI',
            value: config.value || 0,
            change: config.change || 0,
            changeType: config.changeType || 'neutral', // positive, negative, neutral
            icon: config.icon || '📊',
            apiEndpoint: config.apiEndpoint || null,
            ...config
        };
        this.chart = null; // For sparkline if needed
    }

    /**
     * Initialize the KPI card
     */
    async init() {
        // Try to find container by ID or use the provided container element
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            // Try finding by data-id attribute
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) {
            console.warn(`KPI container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API if apiEndpoint is provided
        if (this.config.apiEndpoint) {
            try {
                const response = await fetch(this.config.apiEndpoint);
                const result = await response.json();
                
                // Handle different response formats
                if (result.count !== undefined) {
                    // Stock explorer - count of stocks
                    this.config.value = result.count;
                    this.config.format = null;
                } else if (result.current_value !== undefined) {
                    // Future outcomes - current portfolio value
                    this.config.value = result.current_value;
                    this.config.format = 'currency';
                } else if (result.projected_value && result.projected_value.mean !== undefined) {
                    // Future outcomes - projected value
                    this.config.value = result.projected_value.mean;
                    this.config.format = 'currency';
                } else if (result.risk_metrics && result.risk_metrics.expected_return !== undefined) {
                    // Future outcomes - expected return
                    this.config.value = result.risk_metrics.expected_return;
                    this.config.format = 'percent';
                } else if (result.value !== undefined) {
                    this.config.value = result.value;
                    this.config.change = result.change || 0;
                    this.config.changeType = result.change_type || 'neutral';
                } else if (result.risk_score !== undefined) {
                    // Finance risk_kpis endpoint format
                    this.config.value = result.risk_score;
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.var_95 !== undefined) {
                    // VaR format
                    this.config.value = result.var_95;
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.num_assets !== undefined) {
                    // Asset count format
                    this.config.value = result.num_assets;
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.total_roi !== undefined) {
                    // Marketing Analytics - Total ROI
                    this.config.value = result.total_roi;
                    this.config.format = 'percent';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.synchronization_score !== undefined) {
                    // Marketing Analytics - Omni-Channel Sync Score
                    this.config.value = result.synchronization_score;
                    this.config.format = 'percent';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.model_accuracy !== undefined) {
                    // Marketing Analytics - Predictability Model Accuracy
                    this.config.value = result.model_accuracy;
                    this.config.format = 'percent';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.xgboost && result.xgboost.mae !== undefined) {
                    // Astronomy ML Models - XGBoost MAE
                    this.config.value = result.xgboost.mae;
                    this.config.format = 'number';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.lightgbm && result.lightgbm.mae !== undefined) {
                    // Astronomy ML Models - LightGBM MAE
                    this.config.value = result.lightgbm.mae;
                    this.config.format = 'number';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.comparison && result.comparison.best_mae) {
                    // Astronomy ML Models - Best Model
                    const bestModel = result.comparison.best_mae;
                    this.config.value = bestModel === 'xgboost' ? result.xgboost.mae : result.lightgbm.mae;
                    this.config.format = 'number';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.sharpe_ratio !== undefined) {
                    // Game theory - Nash equilibrium Sharpe ratio
                    this.config.value = result.sharpe_ratio;
                    this.config.format = 'number';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.expected_return !== undefined) {
                    // Game theory - Nash equilibrium expected return
                    this.config.value = result.expected_return;
                    this.config.format = 'percent';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.volatility !== undefined) {
                    // Game theory - Nash equilibrium volatility
                    this.config.value = result.volatility;
                    this.config.format = 'percent';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                } else if (result.n_clusters !== undefined) {
                    // Cluster visualization - number of clusters
                    this.config.value = result.n_clusters;
                    this.config.format = 'number';
                    this.config.change = 0;
                    this.config.changeType = 'neutral';
                }
            } catch (error) {
                console.error('Failed to fetch KPI data:', error);
                // Use default values if fetch fails
                if (this.config.value === undefined) this.config.value = 0;
                if (this.config.change === undefined) this.config.change = 0;
            }
        }

        this.render();
    }

    /**
     * Render the KPI card
     */
    render() {
        // Try multiple ways to find the container
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) return;

        // Try to find value and change elements
        let valueEl = container.querySelector(`#${this.containerId}-value`);
        if (!valueEl) {
            valueEl = container.querySelector('.kpi-value');
        }
        
        let changeEl = container.querySelector(`#${this.containerId}-change`);
        if (!changeEl) {
            changeEl = container.querySelector('.kpi-change');
        }

        const kpiContent = container.querySelector('.kpi-content') || container;

        if (valueEl) {
            valueEl.textContent = this.formatValue(this.config.value);
        }

        if (changeEl) {
            changeEl.textContent = this.formatChange(this.config.change, this.config.changeType);
            changeEl.className = `kpi-change ${this.config.changeType}`;
        }

        this.renderSubtitle(kpiContent);
        this.renderDetails(kpiContent);
    }

    /**
     * Update KPI with new data
     * @param {Object} data - New KPI data
     */
    update(data) {
        this.config = { ...this.config, ...data };
        this.render();
    }

    /**
     * Format value for display
     * TODO (USER): Add number formatting (thousands, millions, etc.)
     */
    formatValue(value) {
        if (value === null || value === undefined || isNaN(value)) return '--';
        
        // Convert to number if string
        const numValue = typeof value === 'string' ? parseFloat(value) : value;
        if (isNaN(numValue)) return '--';
        
        // Handle different format types
        if (this.config.format === 'percent') {
            return `${numValue.toFixed(1)}%`;
        } else if (this.config.format === 'currency') {
            if (numValue >= 1000000) {
                return `$${(numValue / 1000000).toFixed(2)}M`;
            } else if (numValue >= 1000) {
                return `$${(numValue / 1000).toFixed(2)}K`;
            }
            return `$${numValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
        } else if (this.config.format === 'number') {
            // For small numbers (like MAE), show more decimal places
            if (numValue < 1 && numValue > 0) {
                return numValue.toFixed(4);
            } else if (numValue < 0.01) {
                return numValue.toExponential(2);
            }
            return numValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
        }
        
        // Default formatting - smart number formatting
        if (Math.abs(numValue) >= 1000000) {
            return `${(numValue / 1000000).toFixed(2)}M`;
        } else if (Math.abs(numValue) >= 1000) {
            return `${(numValue / 1000).toFixed(2)}K`;
        } else if (Math.abs(numValue) < 1 && numValue !== 0) {
            return numValue.toFixed(4);
        }
        
        return numValue.toLocaleString(undefined, { maximumFractionDigits: 2 });
    }

    /**
     * Format change indicator
     */
    formatChange(change, type) {
        if (change === 0) return 'No change';
        const sign = change > 0 ? '+' : '';
        return `${sign}${change}%`;
    }

    renderSubtitle(container) {
        let subtitleEl = container.querySelector('.kpi-subtitle');
        if (!this.config.subtitle) {
            if (subtitleEl) {
                subtitleEl.remove();
            }
            return;
        }
        if (!subtitleEl) {
            subtitleEl = document.createElement('div');
            subtitleEl.className = 'kpi-subtitle';
            container.appendChild(subtitleEl);
        }
        subtitleEl.textContent = this.config.subtitle;
    }

    renderDetails(container) {
        let detailsWrapper = container.querySelector('.kpi-details');
        const details = this.config.details;

        if (!details || (!details.summary && (!details.top || details.top.length === 0))) {
            if (detailsWrapper) {
                detailsWrapper.remove();
            }
            return;
        }

        if (!detailsWrapper) {
            detailsWrapper = document.createElement('div');
            detailsWrapper.className = 'kpi-details';
            container.appendChild(detailsWrapper);
        }

        let summaryHtml = '';
        if (details.summary || details.threshold) {
            summaryHtml = `
                <div class="kpi-detail-summary">
                    <span>${details.summary || ''}</span>
                    ${details.threshold ? `<span class="threshold-badge">${details.threshold}</span>` : ''}
                </div>
            `;
        }

        let listHtml = '';
        if (details.top && details.top.length > 0) {
            const items = details.top.map(item => `
                <li class="kpi-detail-item">
                    <span class="detail-name">${item.name || 'Unknown'}</span>
                    <span class="detail-meta">
                        ${item.cluster !== null && item.cluster !== undefined ? `<span class="detail-cluster">C${item.cluster}</span>` : ''}
                        ${item.score !== null && item.score !== undefined ? `<span class="detail-score">${this.formatScore(item.score)}σ</span>` : ''}
                    </span>
                </li>
            `).join('');
            listHtml = `<ul class="kpi-detail-list">${items}</ul>`;
        }

        detailsWrapper.innerHTML = summaryHtml + listHtml;
    }

    formatScore(value) {
        if (typeof value !== 'number') {
            const parsed = parseFloat(value);
            if (isNaN(parsed)) return '--';
            return parsed.toFixed(2);
        }
        return value.toFixed(2);
    }

    /**
     * Add sparkline chart using ECharts
     * TODO (USER): Implement sparkline trend visualization
     */
    addSparkline(data) {
        // TODO (USER): Create small ECharts line chart for trend
        // Example structure:
        // const sparklineContainer = document.createElement('div');
        // sparklineContainer.style.width = '100%';
        // sparklineContainer.style.height = '40px';
        // this.chart = echarts.init(sparklineContainer);
        // this.chart.setOption({...});
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = KPICard;
}

