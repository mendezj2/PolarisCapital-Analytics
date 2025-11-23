/**
 * Leaderboard Component
 * Displays ranked list of items
 * TODO (USER): Implement leaderboard visualization
 */

class Leaderboard {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            title: config.title || 'Leaderboard',
            data: config.data || [],
            apiEndpoint: config.apiEndpoint || null,
            maxItems: config.maxItems || 10,
            ...config
        };
    }

    /**
     * Initialize the leaderboard
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
            console.warn(`Leaderboard container ${this.containerId} not found`);
            return;
        }

        // Fetch data from API
        if (this.config.apiEndpoint) {
            try {
                const response = await fetch(this.config.apiEndpoint);
                const result = await response.json();
                
                // Handle different response formats
                if (result.recommendations && Array.isArray(result.recommendations)) {
                    // Future outcomes recommendations
                    this.config.data = result.recommendations.map((rec, idx) => ({
                        rank: idx + 1,
                        name: rec,
                        value: ''
                    }));
                } else if (result.top_performers && Array.isArray(result.top_performers)) {
                    // Stock explorer - top performers (already sorted)
                    this.config.data = result.top_performers.map((stock, idx) => ({
                        rank: idx + 1,
                        name: stock.ticker,
                        value: `${stock.change_percent > 0 ? '+' : ''}${stock.change_percent.toFixed(2)}%`
                    }));
                } else if (result.stocks && Array.isArray(result.stocks)) {
                    // Stock explorer - top performers by change %
                    const sorted = result.stocks
                        .sort((a, b) => (b.change_percent || 0) - (a.change_percent || 0))
                        .slice(0, 10)
                        .map((stock, idx) => ({
                            rank: idx + 1,
                            name: stock.ticker,
                            value: `${stock.change_percent > 0 ? '+' : ''}${stock.change_percent.toFixed(2)}%`
                        }));
                    this.config.data = sorted;
                } else if (Array.isArray(result)) {
                    this.config.data = result;
                } else if (result.data && Array.isArray(result.data)) {
                    this.config.data = result.data;
                }
            } else if (result.recommendations && Array.isArray(result.recommendations)) {
                // Recommendations format (marketing analytics)
                this.config.items = result.recommendations.map((rec, idx) => ({
                    rank: idx + 1,
                    name: rec,
                    value: 0
                }));
            } catch (error) {
                console.error('Failed to fetch leaderboard data:', error);
            }
        }

        this.render();
    }

    /**
     * Render the leaderboard
     * TODO (USER): Add animations, badges, icons
     */
    render() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = '';

        const sortedData = [...this.config.data]
            .sort((a, b) => (b.value || 0) - (a.value || 0))
            .slice(0, this.config.maxItems);

        sortedData.forEach((item, index) => {
            const leaderboardItem = document.createElement('div');
            leaderboardItem.className = 'leaderboard-item';

            leaderboardItem.innerHTML = `
                <span class="leaderboard-rank">#${index + 1}</span>
                <span class="leaderboard-name">${item.name || 'Unknown'}</span>
                <span class="leaderboard-value">${this.formatValue(item.value)}</span>
            `;

            // TODO (USER): Add special styling for top 3 items
            if (index < 3) {
                leaderboardItem.style.background = 'var(--accent-glow)';
            }

            container.appendChild(leaderboardItem);
        });
    }

    /**
     * Format value for display
     */
    formatValue(value) {
        if (value === null || value === undefined) return '--';
        return value.toLocaleString();
    }

    /**
     * Update leaderboard with new data
     */
    update(data) {
        if (data.data) {
            this.config.data = data.data;
        } else if (Array.isArray(data)) {
            this.config.data = data;
        } else {
            this.config = { ...this.config, ...data };
        }
        this.render();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = Leaderboard;
}

