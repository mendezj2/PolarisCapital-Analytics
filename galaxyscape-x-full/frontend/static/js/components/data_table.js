/**
 * Data Table Component
 * Displays tabular data with sorting and filtering
 * TODO (USER): Implement table functionality (sorting, pagination, filtering)
 */

class DataTable {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            columns: config.columns || [],
            data: config.data || [],
            apiEndpoint: config.apiEndpoint || null,
            pageSize: config.pageSize || 10,
            sortable: config.sortable !== false,
            ...config
        };
        this.currentPage = 1;
        this.sortColumn = null;
        this.sortDirection = 'asc';
    }

    /**
     * Initialize the data table
     */
    async init() {
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) {
            console.warn(`Data table container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API
        if (this.config.apiEndpoint) {
            try {
                // Check if this is a POST endpoint
                let response;
                if (this.config.apiEndpoint.includes('/stock/compare')) {
                    // Get symbols from stock-symbol-input component
                    let symbols = [];
                    if (typeof StockSymbolInput !== 'undefined') {
                        const component = window.layoutManager?.components?.get('stock-symbol-input');
                        if (component && typeof component.getSymbols === 'function') {
                            symbols = component.getSymbols();
                        }
                    }
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
                if (result.data && Array.isArray(result.data)) {
                    // star_table format
                    this.config.data = result.data;
                    if (result.columns) {
                        this.config.columns = result.columns.map(col => ({ key: col, label: col }));
                    } else if (result.data.length > 0) {
                        this.config.columns = Object.keys(result.data[0]).map(key => ({ key, label: key }));
                    }
                } else if (result.columns && result.data) {
                    // dashboard/trends format
                    this.config.columns = result.columns.map(col => ({ key: col, label: col }));
                    this.config.data = result.data.map(row => {
                        const obj = {};
                        result.columns.forEach((col, idx) => {
                            obj[col] = row[idx];
                        });
                        return obj;
                    });
                } else if (result.events) {
                    // events format
                    this.config.data = result.events;
                    this.config.columns = [
                        { key: 'timestamp', label: 'Timestamp' },
                        { key: 'ticker', label: 'Ticker' },
                        { key: 'type', label: 'Type' },
                        { key: 'severity', label: 'Severity' },
                        { key: 'description', label: 'Description' }
                    ];
                } else if (result.risk_scores) {
                    // stream/risk format
                    this.config.data = result.risk_scores;
                    this.config.columns = [
                        { key: 'ticker', label: 'Ticker' },
                        { key: 'risk_score', label: 'Risk Score' },
                        { key: 'volatility', label: 'Volatility' },
                        { key: 'anomaly_score', label: 'Anomaly Score' },
                        { key: 'timestamp', label: 'Timestamp' }
                    ];
                } else if (result.stocks && Array.isArray(result.stocks)) {
                    // Stock explorer format
                    this.config.data = result.stocks;
                    this.config.columns = [
                        { key: 'ticker', label: 'Ticker' },
                        { key: 'name', label: 'Name' },
                        { key: 'current_price', label: 'Price', format: 'currency' },
                        { key: 'change_percent', label: 'Change %', format: 'percent' },
                        { key: 'market_cap', label: 'Market Cap', format: 'number' },
                        { key: 'pe_ratio', label: 'P/E' },
                        { key: 'dividend_yield', label: 'Div Yield %', format: 'percent' },
                        { key: 'sector', label: 'Sector' }
                    ];
                } else if (result.payoff_matrix && typeof result.payoff_matrix === 'object') {
                    // Game theory - Prisoner's Dilemma payoff matrix
                    const payoffs = result.payoff_matrix;
                    const nash = result.nash_equilibrium || '';
                    this.config.data = [
                        {
                            strategy: 'Player 1: Cooperate',
                            cooperate: payoffs.cooperate_cooperate || 0,
                            defect: payoffs.cooperate_defect || 0,
                            nash: nash.includes('cooperate') && nash.includes('cooperate') ? '✓' : ''
                        },
                        {
                            strategy: 'Player 1: Defect',
                            cooperate: payoffs.defect_cooperate || 0,
                            defect: payoffs.defect_defect || 0,
                            nash: nash.includes('defect') && nash.includes('defect') ? '✓' : ''
                        }
                    ];
                    this.config.columns = [
                        { key: 'strategy', label: 'Strategy' },
                        { key: 'cooperate', label: 'If Other Cooperates', format: 'number' },
                        { key: 'defect', label: 'If Other Defects', format: 'number' },
                        { key: 'nash', label: 'Nash Equilibrium' }
                    ];
                } else if (result.scenarios && Array.isArray(result.scenarios)) {
                    // Future outcomes scenarios format
                    this.config.data = result.scenarios;
                    this.config.columns = [
                        { key: 'name', label: 'Scenario' },
                        { key: 'value', label: 'Projected Value', format: 'currency' },
                        { key: 'probability', label: 'Probability', format: 'percent' },
                        { key: 'description', label: 'Description' }
                    ];
                } else if (result.in_store_assets && Array.isArray(result.in_store_assets)) {
                    // Signage evaluation format
                    this.config.data = result.in_store_assets;
                    this.config.columns = [
                        { key: 'asset_id', label: 'Asset ID' },
                        { key: 'location', label: 'Location' },
                        { key: 'roi', label: 'ROI %', format: 'number' },
                        { key: 'cost', label: 'Cost', format: 'currency' },
                        { key: 'revenue', label: 'Revenue', format: 'currency' },
                        { key: 'impressions', label: 'Impressions', format: 'number' }
                    ];
                } else if (result.channel_expansion && typeof result.channel_expansion === 'object') {
                    // Channel expansion format (Lowe's & Home Depot)
                    const expansionData = [];
                    if (result.channel_expansion.lowes) {
                        expansionData.push({
                            channel: 'Lowe\'s',
                            current_penetration: result.channel_expansion.lowes.current_penetration,
                            target_penetration: result.channel_expansion.lowes.target_penetration,
                            growth_potential: result.channel_expansion.lowes.growth_potential,
                            strategy: result.channel_expansion.lowes.strategy
                        });
                    }
                    if (result.channel_expansion.home_depot) {
                        expansionData.push({
                            channel: 'Home Depot',
                            current_penetration: result.channel_expansion.home_depot.current_penetration,
                            target_penetration: result.channel_expansion.home_depot.target_penetration,
                            growth_potential: result.channel_expansion.home_depot.growth_potential,
                            strategy: result.channel_expansion.home_depot.strategy
                        });
                    }
                    this.config.data = expansionData;
                    this.config.columns = [
                        { key: 'channel', label: 'Channel' },
                        { key: 'current_penetration', label: 'Current %', format: 'percent' },
                        { key: 'target_penetration', label: 'Target %', format: 'percent' },
                        { key: 'growth_potential', label: 'Growth Potential %', format: 'percent' },
                        { key: 'strategy', label: 'Strategy' }
                    ];
                }
            } catch (error) {
                console.error('Failed to fetch table data:', error);
                // Still render with empty/default data
            }
        }

        // Ensure we have at least empty arrays for rendering
        if (!this.config.data) this.config.data = [];
        if (!this.config.columns) this.config.columns = [];

        this.render();
    }

    /**
     * Render the data table
     */
    render() {
        let container = document.getElementById(this.containerId);
        if (!container && this.config.container) {
            container = this.config.container;
        }
        if (!container) {
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) return;

        // Find or create table
        let table = container.querySelector('table');
        if (!table) {
            table = document.createElement('table');
            table.className = 'data-table';
            container.appendChild(table);
        }

        // Create thead if needed
        let thead = table.querySelector('thead');
        if (!thead) {
            thead = document.createElement('thead');
            table.appendChild(thead);
        }

        // Create tbody if needed
        let tbody = table.querySelector('tbody');
        if (!tbody) {
            tbody = document.createElement('tbody');
            table.appendChild(tbody);
        }

        // Clear existing content
        thead.innerHTML = '';
        tbody.innerHTML = '';

        // Ensure we have valid data structures
        if (!this.config.data) this.config.data = [];
        if (!this.config.columns) this.config.columns = [];
        
        if (this.config.columns.length === 0 || this.config.data.length === 0) {
            container.innerHTML = '<div style="padding: 2rem; text-align: center; color: var(--text-secondary);">No data available. Please upload data or ensure CSV files are present.</div>';
            return;
        }

        // Render header
        const headerRow = document.createElement('tr');
        this.config.columns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.label || col.key;
            th.style.color = 'var(--text-primary)';
            th.style.fontWeight = '700';
            th.style.fontSize = '0.875rem';
            th.style.padding = '0.75rem';
            th.style.borderBottom = '2px solid var(--border-color)';
            if (this.config.sortable) {
                th.style.cursor = 'pointer';
                th.addEventListener('click', () => this.sort(col.key));
            }
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Render rows
        const paginatedData = this.getPaginatedData();
        paginatedData.forEach(row => {
            const tr = document.createElement('tr');
            this.config.columns.forEach(col => {
                const td = document.createElement('td');
                const value = row[col.key];
                td.style.color = 'var(--text-primary)';
                td.style.fontSize = '0.875rem';
                td.style.padding = '0.75rem';
                td.style.borderBottom = '1px solid var(--border-color)';
                if (value !== null && value !== undefined) {
                    td.textContent = typeof value === 'number' ? value.toLocaleString() : String(value);
                } else {
                    td.textContent = '--';
                    td.style.color = 'var(--text-secondary)';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }

    /**
     * Get paginated data
     * TODO (USER): Implement proper pagination logic
     */
    getPaginatedData() {
        const start = (this.currentPage - 1) * this.config.pageSize;
        const end = start + this.config.pageSize;
        return this.config.data.slice(start, end);
    }

    /**
     * Update table with new data
     */
    update(data) {
        this.config = { ...this.config, ...data };
        this.render();
    }

    /**
     * Sort table by column
     */
    sort(column) {
        if (this.sortColumn === column) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else {
            this.sortColumn = column;
            this.sortDirection = 'asc';
        }

        this.config.data.sort((a, b) => {
            const aVal = a[column];
            const bVal = b[column];
            
            if (aVal === null || aVal === undefined) return 1;
            if (bVal === null || bVal === undefined) return -1;
            
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return this.sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
            }
            
            const aStr = String(aVal).toLowerCase();
            const bStr = String(bVal).toLowerCase();
            
            if (this.sortDirection === 'asc') {
                return aStr.localeCompare(bStr);
            } else {
                return bStr.localeCompare(aStr);
            }
        });
        
        this.render();
    }

    /**
     * Update table with new data
     */
    update(data) {
        if (data.columns) this.config.columns = data.columns;
        if (data.data) this.config.data = data.data;
        this.render();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = DataTable;
}

