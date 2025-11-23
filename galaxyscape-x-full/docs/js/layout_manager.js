/**
 * Layout Manager
 * Manages dashboard layout, component initialization, and dashboard switching
 * TODO (USER): Implement dashboard persistence, drag-and-drop, customization
 */

class LayoutManager {
    constructor() {
        this.currentDomain = 'astronomy';
        this.currentDashboard = 'overview';
        this.components = new Map();
        this.dashboardConfigs = {
            astronomy: {},
            finance: {}
        };
        this.currentFilters = {};
        this.globalFilterPanel = null;
        this.datasetContexts = {};
        this.filterPanelV2 = null;
        this.defaultSummaries = {};
    }

    /**
     * Initialize the layout manager
     */
    async init() {
        this.setupEventListeners();
        // Set initial background
        this.updateDashboardBackground(this.currentDomain, this.currentDashboard);
        await this.ensureDefaultDataset(this.currentDomain);
        await this.loadDashboard(this.currentDomain, this.currentDashboard);
        
        // TODO (USER): Load saved dashboard layouts from localStorage
        // TODO (USER): Initialize component resize handlers
    }

    /**
     * Setup event listeners for dashboard navigation
     */
    setupEventListeners() {
        // Dashboard link clicks - use event delegation for better handling
        document.addEventListener('click', (e) => {
            const link = e.target.closest('.dashboard-link');
            if (link) {
                e.preventDefault();
                const dashboard = link.dataset.dashboard;
                const domain = this.currentDomain;
                if (dashboard) {
                this.loadDashboard(domain, dashboard);
                }
            }
        });

        // Domain toggle - use event delegation
        document.addEventListener('click', (e) => {
            const btn = e.target.closest('.domain-btn');
            if (btn && btn.dataset.domain) {
                e.preventDefault();
                e.stopPropagation();
                const domain = btn.dataset.domain;
                this.switchDomain(domain);
            }
            
            // Also handle clicks on images inside domain buttons
            const img = e.target.closest('img');
            if (img && img.closest('.domain-btn')) {
                const btn = img.closest('.domain-btn');
                if (btn && btn.dataset.domain) {
                    e.preventDefault();
                    e.stopPropagation();
                    const domain = btn.dataset.domain;
                    this.switchDomain(domain);
                }
            }
        });

        // Window resize handler for responsive layouts
        window.addEventListener('resize', () => this.handleResize());

        const toggleFiltersBtn = document.getElementById('toggle-filters-btn');
        if (toggleFiltersBtn) {
            toggleFiltersBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.toggleFilterMode();
            });
        }
    }

    /**
     * Switch between Astronomy and Finance domains
     */
    switchDomain(domain) {
        if (domain === this.currentDomain) return;

        this.currentDomain = domain;
        
        // Update UI
        document.body.setAttribute('data-mode', domain);
        document.querySelectorAll('.domain-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.domain === domain);
        });

        // Show/hide dashboard groups
        document.querySelectorAll('.dashboard-group').forEach(group => {
            group.style.display = group.dataset.domain === domain ? 'block' : 'none';
        });

        // Toggle astronomy/finance specific elements
        this.toggleDomainElements(domain);

        // Update theme CSS
        const astroTheme = document.getElementById('theme-astronomy');
        const financeTheme = document.getElementById('theme-finance');
        if (domain === 'astronomy') {
            astroTheme.disabled = false;
            financeTheme.disabled = true;
        } else {
            astroTheme.disabled = true;
            financeTheme.disabled = false;
        }

        // Load first dashboard of new domain
        const firstLink = document.querySelector(`[data-domain="${domain}"] .dashboard-link`);
        if (firstLink && firstLink.dataset.dashboard) {
            this.loadDashboard(domain, firstLink.dataset.dashboard);
        } else {
            // Fallback to default dashboard
            const defaultDashboard = domain === 'astronomy' ? 'overview' : 'risk';
            this.loadDashboard(domain, defaultDashboard);
        }

        // TODO (USER): Add smooth transition animations
    }

    /**
     * Toggle visibility of domain-specific elements
     */
    toggleDomainElements(domain) {
        // Show/hide astronomy-only elements
        document.querySelectorAll('.astronomy-only').forEach(el => {
            el.style.display = domain === 'astronomy' ? '' : 'none';
        });

        // Show/hide finance-only elements
        document.querySelectorAll('.finance-only').forEach(el => {
            el.style.display = domain === 'finance' ? '' : 'none';
        });

        // Update table headers
        const tableHeaders = document.querySelectorAll('thead tr');
        tableHeaders.forEach(tr => {
            if (tr.classList.contains('astronomy-only')) {
                tr.style.display = domain === 'astronomy' ? '' : 'none';
            } else if (tr.classList.contains('finance-only')) {
                tr.style.display = domain === 'finance' ? '' : 'none';
            }
        });
    }

    /**
     * Load a specific dashboard
     */
    async loadDashboard(domain, dashboardName) {
        this.currentDomain = domain;
        this.currentDashboard = dashboardName;
        await this.ensureDefaultDataset(domain);

        // Dispatch dashboard changed event
        document.dispatchEvent(new CustomEvent('dashboard-changed', {
            detail: { domain, dashboard: dashboardName }
        }));

        // Toggle domain-specific elements
        this.toggleDomainElements(domain);

        // Update active dashboard link
        document.querySelectorAll('.dashboard-link').forEach(link => {
            link.classList.toggle('active', link.dataset.dashboard === dashboardName);
        });

        // Update dashboard title
        const titleEl = document.getElementById('dashboard-title');
        if (titleEl) {
            titleEl.textContent = this.getDashboardTitle(domain, dashboardName);
        }
        
        // Setup help button
        this.setupHelpButton(domain, dashboardName);

        // Show/hide report button based on domain
        const reportBtn = document.getElementById('generate-report-btn');
        if (reportBtn) {
            reportBtn.style.display = domain === 'finance' ? 'flex' : 'none';
        }

        // Update background image based on dashboard
        this.updateDashboardBackground(domain, dashboardName);

        // Clear existing components
        this.destroyComponents();

        // Fetch dataset context for capability mode
        // Learning note: dataset context tells us which dashboards/components
        // have enough columns to render, preventing empty charts.
        const datasetContext = await this.getDatasetContext(domain);
        this.applyDashboardAvailability(datasetContext, domain);
        this.showCapabilityBanner(datasetContext, dashboardName);

        // Get dashboard configuration
        const dashboardConfig = this.getDashboardConfig(domain, dashboardName);
        
        if (dashboardConfig) {
            // Render dashboard layout from configuration
            this.renderDashboardLayout(dashboardConfig, datasetContext);
        } else {
            // Fallback: initialize components based on data attributes
        this.initializeComponents();
        }
    }

    /**
     * Get dashboard configuration from config files
     */
    getDashboardConfig(domain, dashboardName) {
        if (domain === 'astronomy' && typeof AstronomyDashboards !== 'undefined') {
            return AstronomyDashboards[dashboardName];
        } else if (domain === 'finance' && typeof FinanceDashboards !== 'undefined') {
            return FinanceDashboards[dashboardName];
        }
        return null;
    }

    /**
    * Fetch dataset context for the current domain (cached per domain)
    */
    async getDatasetContext(domain) {
        if (this.datasetContexts[domain]) {
            return this.datasetContexts[domain];
        }
        try {
            const response = await fetch(`/api/${domain}/context`);
            if (response.ok) {
                const ctx = await response.json();
                this.datasetContexts[domain] = ctx;
                return ctx;
            }
        } catch (error) {
            console.error('Failed to load dataset context', error);
        }
        const fallback = {
            domain,
            capabilityMode: 'explanation',
            schema: { columns: [], dtypes: {}, row_count: 0 },
            featureMapping: {},
            metricAvailability: {},
            missingByDashboard: {}
        };
        this.datasetContexts[domain] = fallback;
        return fallback;
    }

    async ensureDefaultDataset(domain) {
        if (this.defaultSummaries[domain]) {
            return this.defaultSummaries[domain];
        }
        const cached = window.defaultDatasetSummary && window.defaultDatasetSummary[domain];
        if (cached) {
            this.defaultSummaries[domain] = cached;
            return cached;
        }
        try {
            const resp = await fetch(`/api/${domain}/default`);
            if (resp.ok) {
                const data = await resp.json();
                this.defaultSummaries[domain] = data;
                window.defaultDatasetSummary = window.defaultDatasetSummary || {};
                window.defaultDatasetSummary[domain] = data;
                return data;
            }
        } catch (error) {
            console.warn(`Failed to fetch ${domain} default dataset summary`, error);
        }
        return null;
    }

    /**
     * Grey out dashboards with insufficient data
     */
    applyDashboardAvailability(context, domain) {
        const availability = context?.metricAvailability || {};
        const links = document.querySelectorAll(`.dashboard-group[data-domain="${domain}"] .dashboard-link`);
        links.forEach(link => {
            const dash = link.dataset.dashboard;
            const state = availability[dash];
            const isAvailable = !state || state.available || (state.coverage ?? 0) > 0.4;
            link.classList.toggle('disabled', !isAvailable);
            if (state && state.missing && state.missing.length) {
                link.title = `Requires: ${state.missing.join(', ')}`;
            } else {
                link.removeAttribute('title');
            }
        });
    }

    /**
     * Show a capability banner when in partial/explanation mode
     */
    showCapabilityBanner(context, dashboardName) {
        const banner = document.getElementById('capability-banner');
        if (!banner) return;
        const mode = context?.capabilityMode || 'full';
        const dashState = context?.metricAvailability?.[dashboardName];
        if (mode === 'full') {
            banner.style.display = 'none';
            return;
        }
        const missing = dashState?.missing || context?.missingByDashboard?.[dashboardName] || [];
        banner.innerHTML = `
            <div class="capability-label">${mode.toUpperCase()} MODE</div>
            <div class="capability-copy">
                ${mode === 'partial'
                    ? 'Rendering with available columns. Missing: ' + (missing.join(', ') || 'none')
                    : 'Upload a CSV containing domain fields to unlock full dashboards.'}
            </div>
        `;
        banner.style.display = 'flex';
    }

    isComponentAvailable(dashboardName, compConfig, datasetContext = {}) {
        const dashState = datasetContext?.metricAvailability?.[dashboardName];
        if (!dashState) return true;
        if (dashState.available) return true;
        
        const missingSet = new Set(dashState.missing || []);
        const requirements = compConfig.requiredFeatures || [];
        if (requirements.length) {
            const missingForComponent = requirements.filter(f => missingSet.has(f));
            if (missingForComponent.length === requirements.length) {
                return false;
            }
        }
        
        return (dashState.coverage || 0) >= 0.4;
    }

    createPlaceholderComponent(compConfig, datasetContext = {}) {
        const dashState = datasetContext?.metricAvailability?.[this.currentDashboard] || {};
        const dashboardMissing = dashState.missing || [];
        const needed = compConfig.requiredFeatures || [];
        const missing = Array.from(new Set([...(dashboardMissing || []), ...needed]));
        const card = document.createElement('div');
        card.className = 'chart-card placeholder-card';
        card.innerHTML = `
            <div class="card-header">
                <h3>${compConfig.title || compConfig.id}</h3>
                <button class="component-help-btn" title="Explain missing data">?</button>
            </div>
            <div class="card-body empty-state">
                <div class="empty-state-icon">⚠️</div>
                <div class="empty-state-text">Insufficient data for this component.</div>
                <div class="empty-state-subtext">${missing.length ? 'Missing: ' + missing.join(', ') : 'Upload matching columns to populate this view.'}</div>
            </div>
        `;
        const helpBtn = card.querySelector('.component-help-btn');
        helpBtn?.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (window.explainPanel) {
                window.explainPanel.show({
                    title: compConfig.title || compConfig.id,
                    purpose: 'This card is hidden because the dataset does not expose the expected columns.',
                    fields: missing,
                    significance: 'Add the missing columns to unlock the full visualization.',
                    missing
                });
            }
        });
        return card;
    }

    showExplainForComponent(compConfig) {
        const domain = this.currentDomain || 'astronomy';
        const dashboard = this.currentDashboard || 'overview';
        const context = this.datasetContexts[domain] || {};
        const missing = context?.metricAvailability?.[dashboard]?.missing || [];
        const featureMap = context?.featureMapping || {};

        const purposeLookup = {
            'chart-light-curve': 'Plots brightness vs time to reveal rotation period.',
            'chart-age-distribution': 'Shows the distribution of ML-predicted stellar ages.',
            'chart-risk-trends': 'Displays volatility and VaR computed from historical price data.',
            'chart-projection-distribution': 'Monte Carlo simulation showing possible portfolio trajectories.',
            'chart-risk-breakdown': 'Breaks down risk contribution by asset.',
            'network-streaming': 'Correlation network showing how assets co-move in real time.',
            'chart-age-trends': 'Regression of age predictions against observed features.',
            'chart-scenarios': 'Scenario bars comparing optimistic, base, and downside paths.'
        };

        const fieldsByType = {
            'line-chart': ['time', 'value'],
            'bar-chart': ['category', 'value'],
            'pie-chart': ['category', 'share'],
            'network-graph': ['nodes', 'edges', 'weight'],
            'streaming-chart': ['timestamp', 'value'],
            'data-table': ['rows', 'columns']
        };

        const explanation = {
            title: compConfig.title || compConfig.id,
            purpose: purposeLookup[compConfig.id] || purposeLookup[dashboard] || `Explains how ${compConfig.title || compConfig.id} is derived from the dataset.`,
            fields: fieldsByType[compConfig.type] || Object.values(featureMap).slice(0, 4),
            model: dashboard.includes('ml') ? 'Gradient Boosted Regression / LightGBM (demo)' : 'Descriptive analytics',
            equation: compConfig.type === 'line-chart' ? 'y = f(x) where f is fitted using historical observations' : 'N/A',
            significance: dashboard === 'future-outcomes'
                ? 'Helps interpret expected value, volatility bands, and tail risk for portfolios.'
                : 'Use this visual to spot trends, outliers, or correlations quickly.',
            missing
        };

        if (window.explainPanel) {
            window.explainPanel.show(explanation);
        } else if (window.componentHelp) {
            window.componentHelp.showComponentHelp(domain, compConfig.type, compConfig.id, compConfig.title || compConfig.id);
        }
    }

    /**
     * Render dashboard layout from configuration
     */
    renderDashboardLayout(config, datasetContext = {}) {
        const dashboardGrid = document.getElementById('dashboard-grid');
        if (!dashboardGrid) return;

        // Clear existing content
        dashboardGrid.innerHTML = '';

        // Render compact filter ribbon outside the grid
        this.renderGlobalFilters(config.globalFilters || [], datasetContext);

        if (!config.components || config.components.length === 0) {
            dashboardGrid.innerHTML = '<div class="grid-row"><div class="kpi-card"><div class="card-header"><h3>No Components</h3></div><div class="card-body"><p>This dashboard has no components configured.</p></div></div></div>';
            return;
        }

        // Group components by row
        const rows = {};
        config.components.forEach(comp => {
            const row = comp.position?.row || 1;
            if (!rows[row]) {
                rows[row] = [];
            }
            rows[row].push(comp);
        });

        // Sort rows
        const sortedRows = Object.keys(rows).sort((a, b) => parseInt(a) - parseInt(b));

        // Render each row
        sortedRows.forEach(rowKey => {
            const rowComponents = rows[rowKey];
            const rowDiv = document.createElement('div');
            rowDiv.className = 'grid-row';
            
            // Check if any component in this row is fullWidth
            const hasFullWidth = rowComponents.some(comp => comp.position?.fullWidth);
            if (hasFullWidth) {
                rowDiv.classList.add('full-width');
            }

            rowComponents.forEach(comp => {
                const componentAvailable = this.isComponentAvailable(this.currentDashboard, comp, datasetContext);
                if (!componentAvailable) {
                    const placeholder = this.createPlaceholderComponent(comp, datasetContext);
                    rowDiv.appendChild(placeholder);
                } else {
                    const componentEl = this.createComponentElement(comp);
                    if (componentEl) {
                        rowDiv.appendChild(componentEl);
                    }
                }
            });

            dashboardGrid.appendChild(rowDiv);
        });

        // Initialize all components after rendering
        setTimeout(() => {
        this.initializeComponents();
            this.initializeFilters(config);
        }, 100);
    }

    /**
     * Render global filter panel at the top of dashboard
     */
    renderGlobalFilters(filters, datasetContext = {}) {
        const ribbon = document.getElementById('filter-ribbon');
        const drawer = document.getElementById('filter-drawer');
        if (!ribbon || !drawer) return;

        if (!filters || filters.length === 0) {
            ribbon.style.display = 'none';
            drawer.style.display = 'none';
            return;
        }

        const enrichedFilters = this.populateFilterDefaults(filters);

        if (typeof FilterPanelV2 !== 'undefined') {
            if (!this.filterPanelV2) {
                this.filterPanelV2 = new FilterPanelV2({
                    ribbonId: 'filter-ribbon',
                    drawerId: 'filter-drawer',
                    filters: enrichedFilters,
                    onChange: (payload) => this.handleGlobalFilterChange(payload, datasetContext)
                });
                this.filterPanelV2.init();
            } else {
                this.filterPanelV2.updateFilters(enrichedFilters);
            }
        }
    }

    populateFilterDefaults(filters = []) {
        if (!filters || filters.length === 0) return filters;
        const summary = this.defaultSummaries[this.currentDomain] || (window.defaultDatasetSummary || {})[this.currentDomain];
        if (!summary) return filters;
        const featureMap = {
            rotation_min: 'rotation_period',
            rotation_max: 'rotation_period',
            period_min: 'rotation_period',
            period_max: 'rotation_period',
            color_min: 'color_index',
            color_max: 'color_index',
            mass_min: 'mass',
            mass_max: 'mass',
            temperature_min: 'temperature',
            temperature_max: 'temperature',
            luminosity_min: 'luminosity',
            luminosity_max: 'luminosity',
            magnitude_min: 'magnitude',
            magnitude_max: 'magnitude',
            age_min: 'stellar_age',
            age_max: 'stellar_age',
            risk_min: 'risk_score',
            risk_max: 'risk_score',
            volatility_min: 'rolling_vol_21',
            volatility_max: 'rolling_vol_21'
        };

        return filters.map(filter => {
            const enriched = { ...filter };
            if (enriched.options) {
                enriched.options = Array.isArray(enriched.options) ? [...enriched.options] : enriched.options;
            }

            if (enriched.type === 'date-range' && summary.dateRange) {
                enriched.defaultStart = summary.dateRange.start || enriched.defaultStart;
                enriched.defaultEnd = summary.dateRange.end || enriched.defaultEnd;
            }

            if (enriched.name === 'sectors' && Array.isArray(summary.sectors) && summary.sectors.length) {
                enriched.options = summary.sectors.map(value => ({ value, label: value }));
            }

            if (enriched.name === 'tickers' && Array.isArray(summary.tickers) && summary.tickers.length && !enriched.placeholder) {
                enriched.placeholder = `e.g. ${summary.tickers.slice(0, 3).join(', ')}`;
            }

            if (enriched.name === 'clusters' && Array.isArray(summary.clusters) && summary.clusters.length) {
                enriched.options = summary.clusters.map(value => ({ value: value, label: `Cluster ${value}` }));
            }

            const featureKey = featureMap[enriched.name];
            if (featureKey && summary.featureRanges && summary.featureRanges[featureKey]) {
                const stats = summary.featureRanges[featureKey];
                if (enriched.name.endsWith('_min')) {
                    enriched.min = enriched.min ?? stats.min;
                    enriched.max = enriched.max ?? stats.max;
                    enriched.defaultValue = enriched.defaultValue ?? stats.min;
                } else if (enriched.name.endsWith('_max')) {
                    enriched.min = enriched.min ?? stats.min;
                    enriched.max = enriched.max ?? stats.max;
                    enriched.defaultValue = enriched.defaultValue ?? stats.max;
                } else if (enriched.type === 'slider') {
                    enriched.min = enriched.min ?? stats.min;
                    enriched.max = enriched.max ?? stats.max;
                    enriched.defaultValue = enriched.defaultValue ?? stats.median;
                }
            }

            return enriched;
        });
    }

    /**
     * Handle global filter changes - update all components
     * Learning note: global filters are merged then sent to any component
     * that opted into `useFilteredData`, so one interaction refreshes many charts.
     */
    async handleGlobalFilterChange(filters, datasetContext = null) {
        this.currentFilters = { ...this.currentFilters, ...filters };
        
        // Special handling for future outcomes - update portfolio projection
        if (this.currentDashboard === 'future-outcomes') {
            const portfolioInput = this.components.get('portfolio-input');
            if (portfolioInput && portfolioInput.getPortfolio) {
                const portfolio = portfolioInput.getPortfolio();
                if (Object.keys(portfolio).length > 0) {
                    this.updateFutureOutcomes(portfolio);
                    return;
                }
            }
        }
        
        // Update all components that use filtered data
        const dashboardConfig = this.getDashboardConfig(this.currentDomain, this.currentDashboard);
        if (!dashboardConfig) return;

        // Update each component
        dashboardConfig.components.forEach(comp => {
            if (comp.useFilteredData) {
                this.updateComponentWithFilters(comp.id, comp);
            }
        });
    }

    /**
     * Update a component with filtered data
     */
    async updateComponentWithFilters(componentId, compConfig) {
        const component = this.components.get(componentId);
        if (!component) return;

        // Determine data type based on component type
        let dataType = 'chart';
        if (compConfig.type === 'data-table' || compConfig.type === 'table') {
            dataType = 'table';
        } else if (compConfig.type === 'kpi' || compConfig.type === 'gauge') {
            dataType = 'kpi';
        } else if (compConfig.type === 'network-graph') {
            dataType = 'network';
        }

        // Build filter payload - flatten nested objects
        const filterPayload = {
            data_type: dataType,
            dashboard: this.currentDashboard,
            domain: this.currentDomain
        };
        
        Object.keys(this.currentFilters).forEach(key => {
            const value = this.currentFilters[key];
            if (value !== null && value !== undefined) {
                if (typeof value === 'object' && !Array.isArray(value)) {
                    // Flatten date-range and number range objects
                    if (value.start) filterPayload[`${key}_start`] = value.start;
                    if (value.end) filterPayload[`${key}_end`] = value.end;
                    if (value.min !== null && value.min !== undefined) filterPayload[`${key}_min`] = value.min;
                    if (value.max !== null && value.max !== undefined) filterPayload[`${key}_max`] = value.max;
                } else {
                    filterPayload[key] = value;
                }
            }
        });
        
        // Handle date_range special case
        if (this.currentFilters.date_range) {
            if (this.currentFilters.date_range.start) filterPayload.date_start = this.currentFilters.date_range.start;
            if (this.currentFilters.date_range.end) filterPayload.date_end = this.currentFilters.date_range.end;
        }

        // Determine endpoint
        const domain = this.currentDomain;
        const endpoint = `/api/${domain}/get_filtered_data`;

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(filterPayload)
            });

            if (!response.ok) {
                console.error(`Failed to fetch filtered data for ${componentId}`);
                return;
            }

            const data = await response.json();

            // Update component based on type
            if (compConfig.type === 'line-chart' || compConfig.type === 'bar-chart' || compConfig.type === 'pie-chart' || compConfig.type === 'streaming-chart') {
                if (component.update) {
                    component.update(data);
                } else if (component.chart) {
                    // Direct ECharts update with smooth animation
                    const option = {};
                    if (data.xAxis) {
                        option.xAxis = { data: data.xAxis };
                    }
                    if (data.series) {
                        option.series = data.series;
                    } else if (data.data) {
                        option.series = [{ data: data.data }];
                    }
                    if (Object.keys(option).length > 0) {
                        component.chart.setOption(option, true);  // true = notMerge for full update
                    }
                }
            } else if (compConfig.type === 'data-table') {
                if (component.update) {
                    component.update(data);
                } else {
                    // Fallback: update config and re-render
                    component.config.data = data.data || data;
                    component.render();
                }
            } else if (compConfig.type === 'network-graph') {
                if (component.update) {
                    component.update(data);
                } else if (data.nodes && data.edges) {
                    component.config.nodes = data.nodes;
                    component.config.edges = data.edges;
                    component.renderECharts();
                }
            } else if (compConfig.type === 'kpi') {
                if (component.update) {
                    // Extract value from filtered data
                    const value = data.risk_score || data.value || data.num_assets || data.var_95 || 0;
                    const change = data.change || 0;
                    component.update({ value: value, change: change });
                } else {
                    // Fallback: update DOM directly
                    const valueEl = document.querySelector(`[data-id="${componentId}"] .kpi-value`);
                    if (valueEl) {
                        valueEl.textContent = (data.risk_score || data.value || 0).toLocaleString();
                    }
                }
            } else if (compConfig.type === 'gauge') {
                if (component.update) {
                    const value = data.risk_score || data.value || 0;
                    component.update({ value: value });
                } else if (component.chart) {
                    component.chart.setOption({
                        series: [{
                            data: [{
                                value: data.risk_score || data.value || 0,
                                name: compConfig.title
                            }]
                        }]
                    }, true);
                }
            } else if (compConfig.type === 'leaderboard') {
                if (component.update) {
                    component.update({ data: Array.isArray(data) ? data : (data.data || []) });
                } else {
                    component.config.data = Array.isArray(data) ? data : (data.data || []);
                    component.render();
                }
            }
        } catch (error) {
            console.error(`Error updating component ${componentId} with filters:`, error);
        }
    }

    /**
     * Initialize filter panels for dashboard
     */
    initializeFilters(config) {
        if (!config || !config.components) return;

        config.components.forEach(comp => {
            if (comp.filters && comp.filters.length > 0) {
                const filterContainer = document.getElementById(`${comp.id}-filters`);
                if (filterContainer && typeof FilterPanel !== 'undefined') {
                    const filterPanel = new FilterPanel(`${comp.id}-filters`, {
                        filters: comp.filters,
                        onFilterChange: (filters) => {
                            this.handleFilterChange(comp.id, filters);
                        }
                    });
                    filterPanel.init();
                }
            }
        });
    }

    /**
     * Handle filter changes and update components
     */
    handleFilterChange(componentId, filters) {
        // Merge with global filters
        this.currentFilters = { ...this.currentFilters, ...filters };
        
        const component = this.components.get(componentId);
        if (!component) return;

        // Get component config
        const dashboardConfig = this.getDashboardConfig(this.currentDomain, this.currentDashboard);
        if (!dashboardConfig) return;

        const compConfig = dashboardConfig.components.find(c => c.id === componentId);
        if (!compConfig) return;

        // If component uses filtered data, use filtered endpoint
        if (compConfig.useFilteredData) {
            this.updateComponentWithFilters(componentId, compConfig);
            return;
        }

        // Otherwise, use query string approach
        const params = new URLSearchParams();
        Object.keys(filters).forEach(key => {
            const value = filters[key];
            if (value !== null && value !== undefined) {
                if (typeof value === 'object') {
                    if (value.start) params.append(`${key}_start`, value.start);
                    if (value.end) params.append(`${key}_end`, value.end);
                    if (value.min !== null && value.min !== undefined) params.append(`${key}_min`, value.min);
                    if (value.max !== null && value.max !== undefined) params.append(`${key}_max`, value.max);
                } else if (Array.isArray(value)) {
                    value.forEach(v => params.append(key, v));
                } else {
                    params.append(key, value);
                }
            }
        });

        // Update component with new data
        const compEl = document.querySelector(`[data-id="${componentId}"]`);
        if (compEl) {
            const apiEndpoint = compEl.dataset.apiEndpoint;
            if (apiEndpoint) {
                const url = apiEndpoint + (apiEndpoint.includes('?') ? '&' : '?') + params.toString();
                this.refreshComponent(componentId, url);
            }
        }
    }

    /**
     * Refresh a component with new data
     */
    async refreshComponent(componentId, url) {
        const component = this.components.get(componentId);
        if (!component) return;

        try {
            const response = await fetch(url);
            const data = await response.json();
            
            if (component.update) {
                component.update(data);
            } else if (component.chart) {
                // For ECharts components - use setOption for smooth updates
                if (data.xAxis || data.series || data.data) {
                    const option = {};
                    if (data.xAxis) {
                        option.xAxis = { data: data.xAxis };
                    }
                    if (data.series) {
                        option.series = data.series;
                    } else if (data.data) {
                        option.series = [{ data: data.data }];
                    }
                    component.chart.setOption(option, true);  // true = notMerge for full update
                }
            }
        } catch (error) {
            console.error(`Failed to refresh component ${componentId}:`, error);
        }
    }

    /**
     * Create a component element from configuration
     */
    createComponentElement(compConfig) {
        const { type, id, title, apiEndpoint, position, filters } = compConfig;
        
        // Determine card class based on component type
        let cardClass = 'chart-card';
        if (type === 'kpi') cardClass = 'kpi-card';
        else if (type === 'gauge') cardClass = 'gauge-card';
        else if (type === 'data-table') cardClass = 'table-card';
        else if (type === 'leaderboard') cardClass = 'leaderboard-card';
        else if (type === 'network-graph') cardClass = 'network-card';
        else if (type === 'map-card') cardClass = 'map-card';
        else if (type === 'portfolio-input') cardClass = 'table-card';  // Use table-card styling for portfolio input
        else if (type === 'stock-symbol-input') cardClass = 'table-card';  // Use table-card styling for stock symbol input
        else if (type === 'analyze-button') cardClass = 'table-card';  // Use table-card styling for analyze button
        else if (type === 'line-chart' || type === 'bar-chart' || type === 'pie-chart') cardClass = 'chart-card';

        // Create card element
        const card = document.createElement('div');
        card.className = `${cardClass}`;
        card.setAttribute('data-component', type);
        card.setAttribute('data-id', id);
        if (apiEndpoint) {
            card.setAttribute('data-api-endpoint', apiEndpoint);
        }

        // Create card header
        const header = document.createElement('div');
        header.className = 'card-header';
        header.style.display = 'flex';
        header.style.justifyContent = 'space-between';
        header.style.alignItems = 'center';
        
        const headerTitle = document.createElement('h3');
        headerTitle.textContent = title;
        header.appendChild(headerTitle);
        
        // Add help button for component
        const helpBtn = document.createElement('button');
        helpBtn.className = 'component-help-btn';
        helpBtn.title = 'Component Information';
        helpBtn.innerHTML = '<img src="https://cdn-icons-png.flaticon.com/512/1828/1828883.png" alt="Help" style="width: 14px; height: 14px;">';
        helpBtn.style.cssText = `
            padding: 0.25rem 0.5rem;
            background: transparent;
            border: 1px solid var(--border-color);
            border-radius: 50%;
            cursor: pointer;
            opacity: 0.6;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 24px;
            height: 24px;
        `;
        helpBtn.addEventListener('mouseenter', () => {
            helpBtn.style.opacity = '1';
            helpBtn.style.borderColor = 'var(--accent-primary)';
        });
        helpBtn.addEventListener('mouseleave', () => {
            helpBtn.style.opacity = '0.6';
            helpBtn.style.borderColor = 'var(--border-color)';
        });
        helpBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            try {
                this.showExplainForComponent(compConfig);
            } catch (error) {
                console.error('Error showing component help:', error);
            }
        });
        header.appendChild(helpBtn);
        
        card.appendChild(header);

        // Add filter panel if filters are specified
        if (filters && filters.length > 0) {
            const filterContainer = document.createElement('div');
            filterContainer.className = 'filter-panel-container';
            filterContainer.id = `${id}-filters`;
            filterContainer.style.padding = '0 1.5rem 1rem 1.5rem';
            filterContainer.style.borderBottom = '1px solid var(--border-color)';
            card.appendChild(filterContainer);
        }

        // Create card body
        const body = document.createElement('div');
        body.className = 'card-body';

        // Create container based on component type
        let container;
        if (type === 'kpi') {
            container = document.createElement('div');
            container.className = 'kpi-content';
            const value = document.createElement('div');
            value.className = 'kpi-value';
            value.id = `${id}-value`;
            value.textContent = '--';
            const change = document.createElement('div');
            change.className = 'kpi-change';
            change.id = `${id}-change`;
            change.textContent = 'Loading...';
            container.appendChild(value);
            container.appendChild(change);
        } else if (type === 'gauge') {
            container = document.createElement('div');
            container.className = 'gauge-container';
            container.id = `${id}-container`;
        } else if (type === 'network-graph') {
            container = document.createElement('div');
            container.className = 'network-container';
            container.id = `${id}-container`;
        } else if (type === 'data-table') {
            container = document.createElement('div');
            container.className = 'table-container';
            container.id = `${id}-container`;
            const table = document.createElement('table');
            table.className = 'data-table';
            table.id = `${id}-table`;
            container.appendChild(table);
        } else if (type === 'leaderboard') {
            container = document.createElement('div');
            container.className = 'leaderboard-container';
            container.id = `${id}-container`;
        } else if (type === 'map-card') {
            container = document.createElement('div');
            container.className = 'map-container';
            container.id = `${id}-container`;
        } else if (type === 'streaming-chart') {
            container = document.createElement('div');
            container.className = 'chart-container';
            container.id = `${id}-container`;
        } else if (type === 'portfolio-input') {
            container = document.createElement('div');
            container.className = 'portfolio-input-container';
            container.id = `${id}-container`;
        } else if (type === 'stock-symbol-input') {
            container = document.createElement('div');
            container.className = 'stock-symbol-container';
            container.id = `${id}-container`;
        } else {
            // Chart types (line, bar, pie)
            container = document.createElement('div');
            container.className = 'chart-container';
            container.id = `${id}-container`;
        }

        body.appendChild(container);
        card.appendChild(body);

        return card;
    }

    /**
     * Update background image for the current dashboard
     */
    updateDashboardBackground(domain, dashboardName) {
        const dashboardMain = document.querySelector('.dashboard-main');
        if (!dashboardMain) return;

        // Remove existing background classes
        dashboardMain.classList.remove(
            'bg-astronomy-overview', 'bg-astronomy-star-age', 'bg-astronomy-clusters',
            'bg-astronomy-anomalies', 'bg-astronomy-sky-network',
            'bg-finance-risk', 'bg-finance-streaming', 'bg-finance-correlation',
            'bg-finance-portfolio'
        );

        // Add new background class
        const bgClass = `bg-${domain}-${dashboardName}`;
        dashboardMain.classList.add(bgClass);
    }

    /**
     * Initialize all components on current dashboard
     */
    initializeComponents() {
        const components = document.querySelectorAll('[data-component]');
        const dashboardConfig = this.getDashboardConfig(this.currentDomain, this.currentDashboard);
        
        components.forEach(element => {
            const componentType = element.dataset.component;
            const componentId = element.dataset.id || element.id;
            
            if (!componentId) return;

            // Get component config from dashboard config
            let compConfig = null;
            if (dashboardConfig && dashboardConfig.components) {
                compConfig = dashboardConfig.components.find(c => c.id === componentId);
            }

            // Get config from data attributes or component config
            const config = {
                title: compConfig?.title || element.querySelector('.card-header h3')?.textContent || '',
                apiEndpoint: compConfig?.apiEndpoint || element.dataset.apiEndpoint || null,
                useFilteredData: compConfig?.useFilteredData || false,
                ...compConfig
            };

            try {
                let component = null;
                
                switch (componentType) {
                    case 'kpi':
                        if (typeof KPICard !== 'undefined') {
                            const valueEl = element.querySelector(`#${componentId}-value`);
                            const changeEl = element.querySelector(`#${componentId}-change`);
                            if (valueEl || element.querySelector('.kpi-value')) {
                                component = new KPICard(componentId, { ...config, valueElement: valueEl, changeElement: changeEl });
                                
                                // Handle custom rendering for various KPIs
                                if (config.customRender && config.apiEndpoint) {
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            const el = element.querySelector('.kpi-value') || valueEl;
                                            if (!el) return;
                                            
                                            // ML Models Performance KPIs
                                            if (componentId === 'kpi-xgb-mae') {
                                                el.textContent = (data.xgboost?.mae || 0).toFixed(2);
                                            } else if (componentId === 'kpi-lgbm-mae') {
                                                el.textContent = (data.lightgbm?.mae || 0).toFixed(2);
                                            } else if (componentId === 'kpi-best-model') {
                                                const best = data.comparison?.best_mae || 'N/A';
                                                el.textContent = best.toUpperCase();
                                            } else if (componentId === 'kpi-total-roi') {
                                                el.textContent = (data.total_roi || 0).toFixed(1) + '%';
                                            } else if (componentId === 'kpi-sync-score') {
                                                el.textContent = (data.synchronization_score || 0).toFixed(1) + '%';
                                            } else if (componentId === 'kpi-model-accuracy') {
                                                el.textContent = (data.model_accuracy || 0).toFixed(1) + '%';
                                            } else if (componentId === 'kpi-compliant') {
                                                el.textContent = data.compliant || 0;
                                            } else if (componentId === 'kpi-non-compliant') {
                                                el.textContent = data.non_compliant || 0;
                                            } else if (componentId === 'kpi-compliance-rate') {
                                                el.textContent = (data.compliance_rate || 0) + '%';
                                            }
                                        })
                                        .catch(err => console.error('Custom KPI fetch error:', err));
                                } else {
                            component.init();
                                }
                            }
                        }
                        break;
                    case 'bar-chart':
                        if (typeof BarChart !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new BarChart(componentId, { ...config, container });
                            component.init();
                            }
                        }
                        break;
                    case 'line-chart':
                        if (typeof LineChart !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new LineChart(componentId, { ...config, container });
                                
                                // Handle custom rendering for ML models and marketing analytics
                                if (config.customRender && config.apiEndpoint) {
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            // Update chart with custom data
                                            if (component) {
                                                component.update(data);
                                            }
                                        })
                                        .catch(err => console.error('Custom line chart fetch error:', err));
                                } else {
                            component.init();
                                }
                            }
                        }
                        break;
                    case 'pie-chart':
                        if (typeof PieChart !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new PieChart(componentId, { ...config, container });
                                
                                // Handle custom rendering for marketing analytics
                                if (config.customRender && config.apiEndpoint) {
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            // Update chart with custom data
                                            if (component) {
                                                component.update(data);
                                            }
                                        })
                                        .catch(err => console.error('Custom pie chart fetch error:', err));
                                } else {
                            component.init();
                                }
                            }
                        }
                        break;
                    case 'gauge':
                        if (typeof GaugeCard !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new GaugeCard(componentId, { ...config, container });
                            component.init();
                            }
                        }
                        break;
                    case 'data-table':
                        if (typeof DataTable !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new DataTable(componentId, { ...config, container });
                                
                                // Handle custom rendering for various table formats
                                if (config.customRender && config.apiEndpoint) {
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            // Update table with custom data
                                            if (component) {
                                                component.update(data);
                                            }
                                        })
                                        .catch(err => console.error('Custom table fetch error:', err));
                                } else if (componentId === 'table-audit-log' && config.apiEndpoint) {
                                    // Compliance audit log format
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            if (data.audit_log) {
                                                component.config.data = data.audit_log;
                                                component.config.columns = [
                                                    { key: 'date', label: 'Date' },
                                                    { key: 'status', label: 'Status' },
                                                    { key: 'risk_level', label: 'Risk Level' },
                                                    { key: 'description', label: 'Description' }
                                                ];
                                                component.render();
                                            }
                                        });
                                } else {
                            component.init();
                                }
                            }
                        }
                        break;
                    case 'leaderboard':
                        if (typeof Leaderboard !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new Leaderboard(componentId, { ...config, container });
                                
                                // Handle custom rendering for recommendations
                                if (config.customRender && config.apiEndpoint) {
                                    fetch(config.apiEndpoint)
                                        .then(res => res.json())
                                        .then(data => {
                                            // Update leaderboard with custom data
                                            if (component) {
                                                component.update(data);
                                            }
                                        })
                                        .catch(err => console.error('Custom leaderboard fetch error:', err));
                                } else {
                            component.init();
                                }
                            }
                        }
                        break;
                    case 'network-graph':
                        if (typeof NetworkGraph !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new NetworkGraph(componentId, { ...config, container });
                            component.init();
                            }
                        }
                        break;
                    case 'streaming-chart':
                        if (typeof StreamingChart !== 'undefined') {
                            const container = element.querySelector(`#${componentId}-container`);
                            if (container) {
                                component = new StreamingChart(componentId, { ...config, container });
                            component.init();
                            }
                        }
                        break;
                    case 'portfolio-input':
                        if (typeof PortfolioInput !== 'undefined') {
                            component = new PortfolioInput(componentId, {
                                ...config,
                                onPortfolioChange: (portfolio) => {
                                    // Update future outcomes when portfolio changes
                                    if (this.currentDashboard === 'future-outcomes') {
                                        this.updateFutureOutcomes(portfolio);
                                    }
                                }
                            });
                            component.init();
                        }
                        break;
                    case 'stock-symbol-input':
                        if (typeof StockSymbolInput !== 'undefined') {
                            component = new StockSymbolInput(componentId, {
                                ...config,
                                symbols: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'], // Default symbols
                                onSymbolsChange: (symbols) => {
                                    // Update stock explorer when symbols change
                                    if (this.currentDashboard === 'stock-explorer') {
                                        this.updateStockExplorer(symbols);
                                    }
                                }
                            });
                            component.init();
                            
                            // Trigger initial update if we're on stock-explorer dashboard
                            if (this.currentDashboard === 'stock-explorer' && component.getSymbols) {
                                const initialSymbols = component.getSymbols();
                                if (initialSymbols && initialSymbols.length > 0) {
                                    setTimeout(() => {
                                        this.updateStockExplorer(initialSymbols);
                                    }, 500);
                                }
                            }
                        }
                        break;
                    case 'analyze-button':
                        this.setupAnalyzeButton(componentId, config);
                        component = {
                            destroy: () => {
                                // no-op for static button
                            }
                        };
                        break;
                }
                
                if (component) {
                    this.components.set(componentId, component);
                    
                    // If component uses filtered data, fetch with current filters
                    if (config.useFilteredData && Object.keys(this.currentFilters).length > 0) {
                        setTimeout(() => {
                            this.updateComponentWithFilters(componentId, config);
                        }, 200);
                    }
                }
            } catch (error) {
                console.error(`Failed to initialize ${componentType} component ${componentId}:`, error);
            }
        });
    }

    /**
     * Destroy all components
     */
    destroyComponents() {
        this.components.forEach(component => {
            if (component.destroy) {
                component.destroy();
            } else if (component.stopStreaming) {
                component.stopStreaming();
            } else if (component.chart) {
                component.chart.dispose();
            }
        });
        this.components.clear();
        
        // Clear global filter panel
        this.globalFilterPanel = null;
        this.currentFilters = {};
        if (this.filterPanelV2) {
            this.filterPanelV2.current = {};
            this.filterPanelV2.setMode('mini');
        }
        
        // Remove filter panel from DOM
        const filterPanel = document.getElementById('global-filter-panel');
        if (filterPanel) {
            filterPanel.remove();
        }
    }

    /**
     * Update stock explorer dashboard when symbols change
     */
    async updateStockExplorer(symbols) {
        if (!symbols || symbols.length === 0) {
            symbols = ['AAPL', 'MSFT', 'GOOGL']; // Default fallback
        }
        
        // Apply sector filter if set
        const sectorFilter = this.currentFilters.sector;
        const sortBy = this.currentFilters.sort_by || 'market_cap';
        
        try {
            // Update explore endpoint with new symbols
            const exploreUrl = `/api/finance/stock/explore?tickers=${symbols.join(',')}`;
            const exploreResponse = await fetch(exploreUrl);
            
            if (!exploreResponse.ok) {
                console.error('Failed to fetch stock explore data');
                return;
            }
            
            let exploreData = await exploreResponse.json();
            
            // Apply sector filter on frontend if needed
            if (sectorFilter && sectorFilter !== 'all' && exploreData.stocks) {
                exploreData.stocks = exploreData.stocks.filter(stock => stock.sector === sectorFilter);
                exploreData.count = exploreData.stocks.length;
            }
            
            // Apply sorting
            if (exploreData.stocks && sortBy) {
                exploreData.stocks.sort((a, b) => {
                    const aVal = a[sortBy];
                    const bVal = b[sortBy];
                    if (aVal === null || aVal === undefined) return 1;
                    if (bVal === null || bVal === undefined) return -1;
                    if (typeof aVal === 'number' && typeof bVal === 'number') {
                        return bVal - aVal; // Descending
                    }
                    return String(aVal).localeCompare(String(bVal));
                });
            }
            
            // Update compare endpoint with new symbols
            const compareResponse = await fetch('/api/finance/stock/compare', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    tickers: symbols
                })
            });
            
            if (!compareResponse.ok) {
                console.error('Failed to fetch stock compare data');
                return;
            }
            
            const compareData = await compareResponse.json();
            
            // Update all components in stock explorer dashboard
            const dashboardConfig = this.getDashboardConfig('finance', 'stock-explorer');
            if (dashboardConfig) {
                dashboardConfig.components.forEach(comp => {
                    if (comp.id !== 'stock-symbol-input') {
                        const component = this.components.get(comp.id);
                        if (component) {
                            // Determine which data to use based on API endpoint
                            let dataToUse = null;
                            if (comp.apiEndpoint && comp.apiEndpoint.includes('/explore')) {
                                dataToUse = exploreData;
                            } else if (comp.apiEndpoint && comp.apiEndpoint.includes('/compare')) {
                                dataToUse = compareData;
                            }
                            
                            // Update component based on type
                            if (dataToUse) {
                                if (comp.type === 'kpi') {
                                    // Update KPI with count
                                    if (component.update) {
                                        component.update({
                                            value: exploreData.count || 0,
                                            change: 0,
                                            changeType: 'neutral'
                                        });
                                    }
                                } else if (comp.type === 'data-table') {
                                    // Update table with stocks data
                                    if (component.update) {
                                        component.update({
                                            data: exploreData.stocks || [],
                                            columns: [
                                                { key: 'ticker', label: 'Ticker' },
                                                { key: 'name', label: 'Name' },
                                                { key: 'current_price', label: 'Price', format: 'currency' },
                                                { key: 'change_percent', label: 'Change %', format: 'percent' },
                                                { key: 'market_cap', label: 'Market Cap', format: 'number' },
                                                { key: 'pe_ratio', label: 'P/E' },
                                                { key: 'dividend_yield', label: 'Div Yield %', format: 'percent' },
                                                { key: 'sector', label: 'Sector' }
                                            ]
                                        });
                                    }
                                } else if (comp.type === 'bar-chart') {
                                    // Update bar charts with comparison data
                                    if (component.update && compareData.comparison) {
                                        if (comp.id === 'chart-price-comparison') {
                                            // Bar chart for price comparison
                                            component.update({
                                                xAxis: compareData.comparison.map(c => c.ticker),
                                                data: compareData.comparison.map(c => c.current_price)
                                            });
                                        }
                                    }
                                } else if (comp.type === 'line-chart') {
                                    // Update line charts with comparison data
                                    if (component.update && compareData.comparison) {
                                        if (comp.id === 'chart-returns-comparison') {
                                            // Line chart for returns
                                            component.update({
                                                xAxis: ['1M', '3M', '1Y'],
                                                series: compareData.comparison.map(c => ({
                                                    name: c.ticker,
                                                    data: [c.returns_1m || 0, c.returns_3m || 0, c.returns_1y || 0]
                                                }))
                                            });
                                        }
                                    }
                                } else if (comp.type === 'pie-chart') {
                                    // Update pie charts with sector breakdown
                                    if (component.update && exploreData.stocks) {
                                        if (comp.id === 'chart-sector-breakdown') {
                                            // Pie chart for sector breakdown
                                            const sectorCounts = {};
                                            exploreData.stocks.forEach(stock => {
                                                const sector = stock.sector || 'Unknown';
                                                sectorCounts[sector] = (sectorCounts[sector] || 0) + 1;
                                            });
                                            component.update({
                                                data: Object.entries(sectorCounts).map(([name, value]) => ({ name, value }))
                                            });
                                        }
                                    }
                                } else if (comp.type === 'leaderboard') {
                                    // Update leaderboard with top performers
                                    if (component.update && exploreData.stocks) {
                                        const topPerformers = [...exploreData.stocks]
                                            .sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0))
                                            .slice(0, 10)
                                            .map((stock, idx) => ({
                                                name: stock.ticker,
                                                value: stock.change_percent || 0,
                                                change: stock.change_percent
                                            }));
                                        component.update({ data: topPerformers });
                                    }
                                }
                            }
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Error updating stock explorer:', error);
        }
    }

    /**
     * Update future outcomes dashboard when portfolio changes
     */
    async updateFutureOutcomes(portfolio) {
        const timeHorizon = this.currentFilters.time_horizon || 1;
        const confidenceLevel = this.currentFilters.confidence_level || 0.95;
        
        try {
            const response = await fetch('/api/finance/future/outcomes', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    portfolio: portfolio,
                    time_horizon: timeHorizon,
                    confidence_level: confidenceLevel
                })
            });
            
            if (!response.ok) {
                console.error('Failed to fetch future outcomes');
                return;
            }
            
            const data = await response.json();
            
            // Update all components in future outcomes dashboard
            const dashboardConfig = this.getDashboardConfig('finance', 'future-outcomes');
            if (dashboardConfig) {
                dashboardConfig.components.forEach(comp => {
                    if (comp.id !== 'portfolio-input') {
                        const component = this.components.get(comp.id);
                        if (component && component.update) {
                            component.update(data);
                        }
                    }
                });
            }
        } catch (error) {
            console.error('Error updating future outcomes:', error);
        }
    }

    /**
     * Get dashboard title
     */
    getDashboardTitle(domain, dashboardName) {
        // Map dashboard names to titles
        const titleMap = {
            'stock-explorer': 'Stock Explorer & Comparison',
            'future-outcomes': 'Future Outcomes Assessment'
        };
        
        if (titleMap[dashboardName]) {
            return titleMap[dashboardName];
        }
        const titles = {
            astronomy: {
                overview: 'Astronomy Overview',
                'star-explorer': 'Star Explorer',
                'star-age': 'Star Age Analysis',
                'sky-map': 'Sky Map / Projection',
                'light-curve': 'Light Curve / Time Series',
                clusters: 'Cluster Analysis',
                anomalies: 'Anomaly Detection',
                'sky-network': 'Sky Network'
            },
            finance: {
                overview: 'Risk Assessment Dashboard',
                risk: 'Risk Assessment Dashboard',
                streaming: 'Real-Time Risk Analytics',
                correlation: 'Market Correlation Risk',
                portfolio: 'Portfolio Risk Analysis',
                compliance: 'Compliance & Audit Dashboard'
            }
        };
        return titles[domain]?.[dashboardName] || (domain === 'finance' ? 'Risk Assessment Dashboard' : 'Astronomy Overview');
    }

    /**
     * Setup help button for dashboard
     */
    setupHelpButton(domain, dashboardName) {
        const helpBtn = document.getElementById('dashboard-help-btn');
        if (helpBtn) {
            // Remove existing listeners
            const newBtn = helpBtn.cloneNode(true);
            helpBtn.parentNode.replaceChild(newBtn, helpBtn);
            
            newBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                try {
                    if (window.dashboardHelp) {
                        window.dashboardHelp.showHelp(domain, dashboardName);
                    } else {
                        console.warn('DashboardHelp not available');
                    }
                } catch (error) {
                    console.error('Error showing dashboard help:', error);
                }
            });
        }
    }

    /**
     * Setup analyze button for stock explorer
     */
    setupAnalyzeButton(buttonId, config) {
        const buttonEl = document.querySelector(`[data-id="${buttonId}"]`);
        if (!buttonEl) return;

        const cardBody = buttonEl.querySelector('.card-body') || buttonEl;
        cardBody.innerHTML = `
            <div class="analyze-actions-header">
                <span>Model actions</span>
                <span class="badge-soft">Live</span>
            </div>
            <div class="analyze-btn-group">
                <button class="analyze-btn" id="${buttonId}-btn">
                    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" alt="Analyze" style="width: 18px; height: 18px;">
                    Analyze Stocks
                </button>
                <button class="analyze-btn secondary" id="${buttonId}-all-btn">
                    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" alt="Analyze All" style="width: 16px; height: 16px;">
                    Analyze All Symbols
                </button>
            </div>
            <div class="analyze-hint">Uses available price history to compute returns, risk, and recommendations.</div>
        `;

        const btn = document.getElementById(`${buttonId}-btn`);
        const btnAll = document.getElementById(`${buttonId}-all-btn`);
        if (btn) {
            btn.addEventListener('click', async () => {
                await this.analyzeStocks(false);
            });
        }
        if (btnAll) {
            btnAll.addEventListener('click', async () => {
                await this.analyzeStocks(true);
            });
        }
    }

    /**
     * Analyze stocks using ML models
     */
    async analyzeStocks(analyzeAll = false) {
        // Get current symbols from stock symbol input
        const symbolInput = document.querySelector('[data-id="stock-symbol-input"]');
        let symbols = [];
        
        if (symbolInput && typeof StockSymbolInput !== 'undefined') {
            // Try to get symbols from component
            const component = this.components.get('stock-symbol-input');
            if (component && typeof component.getSymbols === 'function') {
                symbols = component.getSymbols();
            }
        }

        if (analyzeAll || symbols.length === 0) {
            try {
                const resp = await fetch('/api/finance/stock/explore');
                const explore = await resp.json();
                symbols = (explore.stocks || []).map(s => s.ticker).slice(0, 12);
            } catch (err) {
                console.error('Failed to load all symbols', err);
            }
        }

        if (symbols.length === 0) {
            showToast('Please add at least one stock symbol to analyze', 'warning');
            return;
        }

        const analyzeBtn = document.getElementById('analyze-stocks-btn-btn');
        const analyzeBtnAll = document.getElementById('analyze-stocks-btn-all-btn');
        [analyzeBtn, analyzeBtnAll].forEach(btn => {
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = `
                    <img src="https://cdn-icons-png.flaticon.com/512/1828/1828840.png" alt="Analyzing" style="width: 18px; height: 18px; animation: spin 1s linear infinite;">
                    Analyzing...
                `;
            }
        });

        try {
            showLoading();
            const response = await fetch('/api/finance/stock/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ tickers: symbols })
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || data.summary || 'Analysis failed');
            }
            
            // Show results in a modal or update dashboard
            this.showAnalysisResults(data);
            showToast('Analysis complete!', 'success');
        } catch (error) {
            console.error('Error analyzing stocks:', error);
            showToast(`Analysis error: ${error.message}`, 'error');
        } finally {
            hideLoading();
            [analyzeBtn, analyzeBtnAll].forEach(btn => {
                if (btn) {
                    btn.disabled = false;
                    btn.innerHTML = `
                        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828843.png" alt="Analyze" style="width: 18px; height: 18px;">
                        ${btn.id.includes('all') ? 'Analyze All Symbols' : 'Analyze Stocks'}
                    `;
                }
            });
        }
    }

    /**
     * Show analysis results
     */
    showAnalysisResults(data) {
        // Create results modal
        const modal = document.createElement('div');
        modal.id = 'analysis-results-modal';
        modal.className = 'help-modal';
        modal.innerHTML = `
            <div class="help-modal-overlay"></div>
            <div class="help-modal-content" style="max-width: 800px;">
                <div class="help-modal-header">
                    <h3>Stock Analysis Results</h3>
                    <button class="help-modal-close" id="analysis-modal-close">
                        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828842.png" alt="Close" style="width: 20px; height: 20px;">
                    </button>
                </div>
                <div class="help-modal-body">
                    ${this.formatAnalysisResults(data)}
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close handlers
        const closeBtn = document.getElementById('analysis-modal-close');
        const overlay = modal.querySelector('.help-modal-overlay');
        
        const closeModal = () => {
            modal.style.animation = 'fadeOut 0.3s ease';
            setTimeout(() => modal.remove(), 300);
        };

        closeBtn?.addEventListener('click', closeModal);
        overlay?.addEventListener('click', closeModal);

        setTimeout(() => {
            modal.style.opacity = '1';
        }, 10);
    }

    /**
     * Format analysis results for display
     */
    formatAnalysisResults(data) {
        let html = '';
        
        if (data.summary) {
            html += `<div class="help-section"><h4>Summary</h4><p>${data.summary}</p></div>`;
        }

        if (data.risk_scores && data.risk_scores.length > 0) {
            html += `<div class="help-section"><h4>Risk Scores</h4><ul>`;
            data.risk_scores.forEach(item => {
                html += `<li><strong>${item.ticker}</strong>: ${item.score.toFixed(2)} (${item.level})</li>`;
            });
            html += `</ul></div>`;
        }

        if (data.predictions && data.predictions.length > 0) {
            html += `<div class="help-section"><h4>Price Predictions</h4><ul>`;
            data.predictions.forEach(item => {
                html += `<li><strong>${item.ticker}</strong>: $${item.predicted_price.toFixed(2)} (${item.confidence}% confidence)</li>`;
            });
            html += `</ul></div>`;
        }

        if (data.recommendations && data.recommendations.length > 0) {
            html += `<div class="help-section"><h4>Recommendations</h4><ul>`;
            data.recommendations.forEach(rec => {
                html += `<li>${rec}</li>`;
            });
            html += `</ul></div>`;
        }

        return html || '<p>Analysis complete. Check the dashboard for updated metrics.</p>';
    }

    /**
     * Handle window resize
     */
    handleResize() {
        this.components.forEach(component => {
            if (component && typeof component.resize === 'function') {
                component.resize();
            }
        });
    }

    /**
     * Toggle filter panel modes (mini -> expanded -> hidden)
     */
    toggleFilterMode() {
        if (this.filterPanelV2 && typeof this.filterPanelV2.cycleMode === 'function') {
            this.filterPanelV2.cycleMode();
        }
    }
}

// Initialize layout manager when DOM is ready
let layoutManager;
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        layoutManager = new LayoutManager();
        layoutManager.init();
        window.layoutManager = layoutManager; // Expose globally
    });
} else {
    layoutManager = new LayoutManager();
    layoutManager.init();
    window.layoutManager = layoutManager; // Expose globally
}
