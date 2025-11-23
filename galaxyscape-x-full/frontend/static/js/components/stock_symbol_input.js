/**
 * Stock Symbol Input Component
 * Allows users to search and add stock/index fund symbols for comparison
 */

class StockSymbolInput {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            onSymbolsChange: config.onSymbolsChange || null,
            maxSymbols: config.maxSymbols || 20,
            ...config
        };
        this.symbols = config.symbols || [];
        this.debounceTimer = null;
    }

    /**
     * Initialize the stock symbol input
     */
    init() {
        let container = document.getElementById(this.containerId);
        if (!container) {
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) {
            // Try to find by data-id attribute and get card-body
            const card = document.querySelector(`[data-id="${this.containerId}"]`);
            if (card) {
                container = card.querySelector('.card-body') || card.querySelector('.portfolio-input-container');
            }
        }
        if (!container) {
            // Try container with id
            container = document.getElementById(`${this.containerId}-container`);
        }
        if (!container) {
            console.warn(`Stock symbol input container ${this.containerId} not found`);
            return;
        }

        this.container = container;
        this.render();
    }

    /**
     * Render stock symbol input form
     */
    render() {
        const container = this.container || document.getElementById(this.containerId) || document.querySelector(`[data-id="${this.containerId}"]`);
        if (!container) return;

        // Find card-body or use container directly
        let cardBody = container.querySelector('.card-body');
        if (!cardBody) {
            if (container.classList.contains('card-body')) {
                cardBody = container;
            } else if (container.classList.contains('portfolio-input-container') || container.classList.contains('stock-symbol-container')) {
                cardBody = container;
            } else {
                cardBody = document.createElement('div');
                cardBody.className = 'card-body';
                container.appendChild(cardBody);
            }
        }

        cardBody.innerHTML = `
            <div style="margin-bottom: 1.5rem;">
                <label style="display: block; margin-bottom: 0.5rem; font-weight: 600; color: var(--text-primary);">
                    Stock/Index Symbols
                </label>
                <div id="${this.containerId}-symbols" style="margin-bottom: 1rem; min-height: 40px;">
                    ${this.renderSymbols()}
                </div>
                <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                    <input 
                        type="text" 
                        id="${this.containerId}-symbol-input" 
                        placeholder="Enter symbol (e.g., AAPL, SPY, ^GSPC)..." 
                        style="flex: 1; padding: 0.75rem; border: 1px solid var(--border-color); border-radius: 4px; background: var(--bg-secondary); color: var(--text-primary); font-size: 0.875rem;"
                        autocomplete="off"
                    >
                    <button 
                        type="button" 
                        id="${this.containerId}-add-btn"
                        style="padding: 0.75rem 1.5rem; background: var(--accent-primary); color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 0.875rem; white-space: nowrap; transition: opacity 0.2s;"
                    >
                        <span id="${this.containerId}-add-btn-text">Add Symbol</span>
                    </button>
                </div>
                <div style="font-size: 0.875rem; color: var(--text-secondary); margin-top: 0.5rem;">
                    <p style="margin: 0.25rem 0;">💡 Enter stock symbols (e.g., AAPL, MSFT, GOOGL) or index symbols (e.g., SPY, QQQ, ^GSPC)</p>
                    <p style="margin: 0.25rem 0;">💡 Separate multiple symbols with commas or add one at a time</p>
                    <p style="margin: 0.25rem 0;">💡 Supports up to ${this.config.maxSymbols} symbols</p>
                </div>
            </div>
        `;

        // Add event listeners
        const addBtn = document.getElementById(`${this.containerId}-add-btn`);
        const symbolInput = document.getElementById(`${this.containerId}-symbol-input`);

        if (addBtn) {
            // Remove existing listeners to prevent duplicates
            const newAddBtn = addBtn.cloneNode(true);
            addBtn.parentNode.replaceChild(newAddBtn, addBtn);
            newAddBtn.addEventListener('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                this.handleAddSymbol();
            });
        }

        // Allow Enter key
        if (symbolInput) {
            // Remove existing listeners to prevent duplicates
            const newSymbolInput = symbolInput.cloneNode(true);
            symbolInput.parentNode.replaceChild(newSymbolInput, symbolInput);
            
            newSymbolInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    e.preventDefault();
                    e.stopPropagation();
                    this.handleAddSymbol();
                }
            });

            // Allow paste with comma-separated values
            newSymbolInput.addEventListener('paste', async (e) => {
                setTimeout(async () => {
                    const value = newSymbolInput.value.trim();
                    if (value.includes(',')) {
                        const symbols = value.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
                        for (const symbol of symbols) {
                            if (symbol && !this.symbols.includes(symbol) && this.symbols.length < this.config.maxSymbols) {
                                await this.addSymbol(symbol);
                                // Small delay between validations
                                await new Promise(resolve => setTimeout(resolve, 300));
                            }
                        }
                        newSymbolInput.value = '';
                    }
                }, 10);
            });
            
            // Real-time search as user types (optional - can be enabled)
            newSymbolInput.addEventListener('input', (e) => {
                const value = e.target.value.trim().toUpperCase();
                // Optional: Add autocomplete/search suggestions here
            });
        }
    }

    /**
     * Handle add symbol button click
     */
    async handleAddSymbol() {
        const symbolInput = document.getElementById(`${this.containerId}-symbol-input`);
        const addBtn = document.getElementById(`${this.containerId}-add-btn`);
        const addBtnText = document.getElementById(`${this.containerId}-add-btn-text`);
        
        if (!symbolInput) return;

        const value = symbolInput.value.trim().toUpperCase();
        if (!value) {
            this.showMessage('Please enter a symbol', 'warning');
            return;
        }

        // Disable button during validation
        if (addBtn) {
            addBtn.disabled = true;
            addBtn.style.opacity = '0.6';
            addBtn.style.cursor = 'not-allowed';
            if (addBtnText) addBtnText.textContent = 'Validating...';
        }

        try {
            // Handle comma-separated values
            if (value.includes(',')) {
                const symbols = value.split(',').map(s => s.trim().toUpperCase()).filter(s => s);
                for (const symbol of symbols) {
                    if (symbol && !this.symbols.includes(symbol) && this.symbols.length < this.config.maxSymbols) {
                        await this.addSymbol(symbol);
                        // Small delay between validations to avoid rate limiting
                        await new Promise(resolve => setTimeout(resolve, 300));
                    }
                }
            } else {
                if (!this.symbols.includes(value) && this.symbols.length < this.config.maxSymbols) {
                    await this.addSymbol(value);
                }
            }
        } finally {
            // Re-enable button
            if (addBtn) {
                addBtn.disabled = false;
                addBtn.style.opacity = '1';
                addBtn.style.cursor = 'pointer';
                if (addBtnText) addBtnText.textContent = 'Add Symbol';
            }
            symbolInput.value = '';
        }
    }

    /**
     * Render current symbols
     */
    renderSymbols() {
        if (this.symbols.length === 0) {
            return '<p style="color: var(--text-secondary); font-size: 0.875rem; padding: 0.75rem; background: var(--bg-secondary); border-radius: 4px; border: 1px dashed var(--border-color);">No symbols added. Enter symbols above to compare.</p>';
        }

        let html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">';
        for (const symbol of this.symbols) {
            html += `
                <div style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.75rem; background: var(--bg-secondary); border-radius: 4px; border: 1px solid var(--border-color);">
                    <span style="font-weight: 600; color: var(--text-primary);">${symbol}</span>
                    <button 
                        type="button" 
                        class="remove-symbol" 
                        data-symbol="${symbol}"
                        style="padding: 0.125rem 0.5rem; background: var(--error-color, #ef4444); color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 0.75rem; font-weight: 600; line-height: 1.2;"
                        title="Remove ${symbol}"
                    >
                        ×
                    </button>
                </div>
            `;
        }
        html += '</div>';

        // Re-attach event listeners after rendering
        setTimeout(() => {
            document.querySelectorAll(`#${this.containerId}-symbols .remove-symbol`).forEach(btn => {
                // Remove existing listeners
                const newBtn = btn.cloneNode(true);
                btn.parentNode.replaceChild(newBtn, btn);
                newBtn.addEventListener('click', (e) => {
                    e.preventDefault();
                    e.stopPropagation();
                    const symbol = newBtn.dataset.symbol || e.target.dataset.symbol;
                    if (symbol) {
                        this.removeSymbol(symbol);
                    }
                });
            });
        }, 100);

        return html;
    }

    /**
     * Validate if a symbol is real/valid
     */
    async validateSymbol(symbol) {
        // Basic format validation first
        if (!symbol || symbol.length < 1 || symbol.length > 10) {
            return { valid: false, error: 'Symbol must be 1-10 characters' };
        }
        
        // Allow common index symbols (^GSPC, ^DJI, etc.)
        if (symbol.startsWith('^')) {
            return { valid: true };
        }
        
        // Check if it's alphanumeric (basic validation)
        if (!/^[A-Z0-9.]+$/.test(symbol)) {
            return { valid: false, error: 'Symbol contains invalid characters' };
        }
        
        // Try to fetch data from API to validate
        try {
            const response = await fetch(`/api/finance/stock/explore?tickers=${symbol}`);
            if (!response.ok) {
                return { valid: false, error: 'Failed to validate symbol' };
            }
            
            const data = await response.json();
            
            // Check if we got data back or errors
            if (data.errors && data.errors.length > 0) {
                const symbolError = data.errors.find(e => e.includes(symbol));
                if (symbolError) {
                    return { valid: false, error: symbolError.split(':')[1]?.trim() || 'Symbol not found' };
                }
            }
            
            // If we have stocks data, check if our symbol is in it
            if (data.stocks && data.stocks.length > 0) {
                const found = data.stocks.find(s => s.ticker === symbol);
                if (found) {
                    return { valid: true, data: found };
                }
            }
            
            // If no errors and no stocks, might be invalid
            if (data.count === 0 && (!data.errors || data.errors.length === 0)) {
                return { valid: false, error: 'Symbol not found or no data available' };
            }
            
            // If we got here and have no errors, assume valid
            return { valid: true };
        } catch (error) {
            // If validation fails, still allow it but warn user
            console.warn(`Could not validate symbol ${symbol}:`, error);
            // For now, allow it - user will see error when data is fetched
            return { valid: true, warning: 'Could not validate, will check when loading data' };
        }
    }

    /**
     * Add a symbol with validation
     */
    async addSymbol(symbol) {
        if (this.symbols.length >= this.config.maxSymbols) {
            this.showMessage(`Maximum ${this.config.maxSymbols} symbols allowed`, 'error');
            return;
        }
        
        if (this.symbols.includes(symbol)) {
            this.showMessage(`${symbol} is already added`, 'warning');
            return;
        }
        
        // Show loading state
        this.showMessage(`Validating ${symbol}...`, 'info');
        
        // Validate symbol
        const validation = await this.validateSymbol(symbol);
        
        if (!validation.valid) {
            this.showMessage(`${symbol}: ${validation.error || 'Invalid symbol'}`, 'error');
            return;
        }
        
        // Add symbol
        this.symbols.push(symbol);
        this.render();
        
        if (validation.warning) {
            this.showMessage(`${symbol} added (${validation.warning})`, 'warning');
        } else {
            this.showMessage(`${symbol} added successfully`, 'success');
        }
        
        if (this.config.onSymbolsChange) {
            this.config.onSymbolsChange(this.symbols);
        }
    }
    
    /**
     * Show a temporary message to the user
     */
    showMessage(message, type = 'info') {
        // Remove existing message
        const existingMsg = document.getElementById(`${this.containerId}-message`);
        if (existingMsg) {
            existingMsg.remove();
        }
        
        // Create message element
        const msgEl = document.createElement('div');
        msgEl.id = `${this.containerId}-message`;
        msgEl.style.cssText = `
            position: absolute;
            top: -40px;
            left: 0;
            right: 0;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.875rem;
            font-weight: 500;
            z-index: 1000;
            animation: slideDown 0.3s ease;
        `;
        
        // Set color based on type
        if (type === 'error') {
            msgEl.style.background = '#ef4444';
            msgEl.style.color = 'white';
        } else if (type === 'warning') {
            msgEl.style.background = '#f59e0b';
            msgEl.style.color = 'white';
        } else if (type === 'success') {
            msgEl.style.background = '#22c55e';
            msgEl.style.color = 'white';
        } else {
            msgEl.style.background = 'var(--bg-secondary)';
            msgEl.style.color = 'var(--text-primary)';
            msgEl.style.border = '1px solid var(--border-color)';
        }
        
        msgEl.textContent = message;
        
        // Find container and append message
        const container = document.getElementById(this.containerId) || 
                         document.querySelector(`[data-id="${this.containerId}"]`);
        if (container) {
            container.style.position = 'relative';
            container.appendChild(msgEl);
            
            // Remove message after 3 seconds
            setTimeout(() => {
                if (msgEl.parentNode) {
                    msgEl.style.animation = 'slideUp 0.3s ease';
                    setTimeout(() => msgEl.remove(), 300);
                }
            }, 3000);
        }
    }

    /**
     * Remove a symbol
     */
    removeSymbol(symbol) {
        this.symbols = this.symbols.filter(s => s !== symbol);
        this.render();
        if (this.config.onSymbolsChange) {
            this.config.onSymbolsChange(this.symbols);
        }
    }

    /**
     * Get current symbols
     */
    getSymbols() {
        return this.symbols;
    }

    /**
     * Set symbols
     */
    setSymbols(symbols) {
        this.symbols = symbols.slice(0, this.config.maxSymbols);
        this.render();
    }

    /**
     * Clear all symbols
     */
    clear() {
        this.symbols = [];
        this.render();
        if (this.config.onSymbolsChange) {
            this.config.onSymbolsChange(this.symbols);
        }
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = StockSymbolInput;
}

