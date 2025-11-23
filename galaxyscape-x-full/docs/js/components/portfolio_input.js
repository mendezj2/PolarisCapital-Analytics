/**
 * Portfolio Input Component
 * Allows users to input their current portfolio holdings for future outcomes analysis
 */

class PortfolioInput {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            onPortfolioChange: config.onPortfolioChange || null,
            ...config
        };
        this.portfolio = config.portfolio || {};
    }

    /**
     * Initialize the portfolio input
     */
    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            container = document.querySelector(`[data-id="${this.containerId}"]`);
        }
        if (!container) {
            console.warn(`Portfolio input container ${this.containerId} not found`);
            return;
        }

        this.render();
    }

    /**
     * Render portfolio input form
     */
    render() {
        const container = this.container || document.getElementById(this.containerId) || document.querySelector(`[data-id="${this.containerId}"]`);
        if (!container) return;

        // Find card-body or use container directly
        let cardBody = container.querySelector('.card-body');
        if (!cardBody) {
            if (container.classList.contains('card-body')) {
                cardBody = container;
            } else if (container.classList.contains('portfolio-input-container')) {
                // If it's the portfolio-input-container, use it directly
                cardBody = container;
            } else {
                // Create card-body if it doesn't exist
                cardBody = document.createElement('div');
                cardBody.className = 'card-body';
                container.appendChild(cardBody);
            }
        }

        cardBody.innerHTML = `
            <div style="margin-bottom: 1.5rem;">
                <label style="display: block; margin-bottom: 0.5rem; font-weight: 600; color: var(--text-primary);">
                    Portfolio Holdings
                </label>
                <div id="${this.containerId}-holdings" style="margin-bottom: 1rem;">
                    ${this.renderHoldings()}
                </div>
                <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
                    <input 
                        type="text" 
                        id="${this.containerId}-ticker" 
                        placeholder="Ticker (e.g., AAPL)" 
                        style="flex: 1; padding: 0.5rem; border: 1px solid var(--border-color); border-radius: 4px; background: var(--bg-secondary); color: var(--text-primary);"
                    >
                    <input 
                        type="number" 
                        id="${this.containerId}-shares" 
                        placeholder="Shares" 
                        min="0" 
                        step="0.01"
                        style="flex: 1; padding: 0.5rem; border: 1px solid var(--border-color); border-radius: 4px; background: var(--bg-secondary); color: var(--text-primary);"
                    >
                    <button 
                        type="button" 
                        id="${this.containerId}-add-btn"
                        style="padding: 0.5rem 1rem; background: var(--accent-primary); color: white; border: none; border-radius: 4px; cursor: pointer; font-weight: 600;"
                    >
                        Add
                    </button>
                </div>
                <div style="font-size: 0.875rem; color: var(--text-secondary);">
                    <p>Enter your current holdings to project future outcomes.</p>
                    <p>Example: AAPL: 10 shares, MSFT: 5 shares</p>
                </div>
            </div>
        `;

        // Add event listeners
        const addBtn = document.getElementById(`${this.containerId}-add-btn`);
        const tickerInput = document.getElementById(`${this.containerId}-ticker`);
        const sharesInput = document.getElementById(`${this.containerId}-shares`);

        if (addBtn) {
            addBtn.addEventListener('click', () => {
                const ticker = tickerInput.value.trim().toUpperCase();
                const shares = parseFloat(sharesInput.value);

                if (ticker && shares > 0) {
                    this.addHolding(ticker, shares);
                    tickerInput.value = '';
                    sharesInput.value = '';
                }
            });
        }

        // Allow Enter key
        if (tickerInput) {
            tickerInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    sharesInput.focus();
                }
            });
        }

        if (sharesInput) {
            sharesInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    addBtn.click();
                }
            });
        }
    }

    /**
     * Render current holdings
     */
    renderHoldings() {
        if (Object.keys(this.portfolio).length === 0) {
            return '<p style="color: var(--text-secondary); font-size: 0.875rem;">No holdings added yet.</p>';
        }

        let html = '<div style="display: flex; flex-direction: column; gap: 0.5rem;">';
        for (const [ticker, shares] of Object.entries(this.portfolio)) {
            html += `
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: var(--bg-secondary); border-radius: 4px; border: 1px solid var(--border-color);">
                    <div>
                        <strong>${ticker}</strong>
                        <span style="color: var(--text-secondary); margin-left: 0.5rem;">${shares} shares</span>
                    </div>
                    <button 
                        type="button" 
                        class="remove-holding" 
                        data-ticker="${ticker}"
                        style="padding: 0.25rem 0.75rem; background: var(--error-color, #ef4444); color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 0.875rem;"
                    >
                        Remove
                    </button>
                </div>
            `;
        }
        html += '</div>';

        // Re-attach event listeners after rendering
        setTimeout(() => {
            document.querySelectorAll(`#${this.containerId}-holdings .remove-holding`).forEach(btn => {
                btn.addEventListener('click', (e) => {
                    const ticker = e.target.dataset.ticker;
                    this.removeHolding(ticker);
                });
            });
        }, 100);

        return html;
    }

    /**
     * Add a holding
     */
    addHolding(ticker, shares) {
        if (this.portfolio[ticker]) {
            this.portfolio[ticker] += shares;
        } else {
            this.portfolio[ticker] = shares;
        }
        this.render();
        if (this.config.onPortfolioChange) {
            this.config.onPortfolioChange(this.portfolio);
        }
    }

    /**
     * Remove a holding
     */
    removeHolding(ticker) {
        delete this.portfolio[ticker];
        this.render();
        if (this.config.onPortfolioChange) {
            this.config.onPortfolioChange(this.portfolio);
        }
    }

    /**
     * Get current portfolio
     */
    getPortfolio() {
        return this.portfolio;
    }

    /**
     * Set portfolio
     */
    setPortfolio(portfolio) {
        this.portfolio = portfolio;
        this.render();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = PortfolioInput;
}

