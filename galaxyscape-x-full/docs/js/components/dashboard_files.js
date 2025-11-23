/**
 * Dashboard Files Component
 * Displays active data files for each dashboard in the sidebar with enhanced UI
 */
class DashboardFilesPanel {
    constructor() {
        this.activeFiles = {};
        this.availableFiles = {};
    }

    /**
     * Initialize the files panel
     */
    async init() {
        this.render();
        await this.refresh();
        
        // Listen for dashboard changes
        document.addEventListener('dashboard-changed', async (e) => {
            await this.refresh(e.detail.domain, e.detail.dashboard);
            this.updateActiveFile(e.detail.domain, e.detail.dashboard);
            this.updatePanel(e.detail.domain, e.detail.dashboard);
        });
    }

    /**
     * Refresh file information for current dashboard
     */
    async refresh(domain = null, dashboard = null) {
        if (!domain || !dashboard) {
            // Get from layout manager if available
            if (window.layoutManager) {
                domain = window.layoutManager.currentDomain;
                dashboard = window.layoutManager.currentDashboard;
            } else {
                return;
            }
        }

        try {
            const response = await fetch(`/api/${domain}/data/files?dashboard=${dashboard}`);
            const data = await response.json();
            
            if (data.active_file) {
                this.activeFiles[`${domain}-${dashboard}`] = data.active_file;
            }
            if (data.available_files) {
                this.availableFiles[`${domain}-${dashboard}`] = data.available_files;
            }
            
            this.updateActiveFile(domain, dashboard);
        } catch (error) {
            console.error('Failed to refresh file info:', error);
        }
    }

    /**
     * Update the active file display for a dashboard
     */
    updateActiveFile(domain, dashboard) {
        const key = `${domain}-${dashboard}`;
        const activeFile = this.activeFiles[key];
        const requiredFeatures = this.getRequiredFeatures(domain, dashboard);
        const coverage = this.evaluateCoverage(activeFile ? activeFile.columns : [], requiredFeatures);
        
        // Find the dashboard link in sidebar
        const dashboardLink = document.querySelector(`[data-dashboard="${dashboard}"]`);
        if (!dashboardLink) return;
        const indicatorSlot = dashboardLink.querySelector('.dashboard-indicator-slot') || dashboardLink;

        // Remove existing file indicator
        const existingIndicator = indicatorSlot.querySelector('.file-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }

        // Add file indicator
        if (activeFile) {
            const indicator = document.createElement('span');
            indicator.className = `file-indicator coverage-${coverage.status}`;
            indicator.title = `Active: ${activeFile.name} (${activeFile.row_count.toLocaleString()} rows)\n${this.getCoverageTooltip(coverage, requiredFeatures)}`;
            indicator.innerHTML = `
                <span class="file-indicator-icon">📄</span>
                <span class="file-indicator-name">${this.truncateFileName(activeFile.name, 18)}</span>
                <span class="coverage-dot ${coverage.status}" aria-hidden="true"></span>
            `;
            indicatorSlot.appendChild(indicator);
        }
    }

    /**
     * Render the files panel in sidebar
     */
    render() {
        const sidebar = document.getElementById('dashboard-sidebar');
        if (!sidebar) return;

        // Check if panel already exists
        let panel = document.getElementById('dashboard-files-panel');
        if (!panel) {
            panel = document.createElement('div');
            panel.id = 'dashboard-files-panel';
            panel.className = 'dashboard-files-panel';
            panel.innerHTML = `
                <div class="files-panel-header">
                    <h3>
                        <span class="files-panel-icon">📊</span>
                        <span>Data Files</span>
                    </h3>
                    <div class="files-panel-actions">
                        <button class="files-default-btn" title="Use default dataset" aria-label="Use default">
                            <span class="default-icon">✨</span>
                        </button>
                        <button class="files-refresh-btn" title="Refresh files" aria-label="Refresh">
                            <span class="refresh-icon">🔄</span>
                        </button>
                    </div>
                </div>
                <div class="files-panel-content" id="files-panel-content">
                    <div class="files-loading">
                        <span class="loading-spinner"></span>
                        <span>Loading files...</span>
                    </div>
                </div>
            `;
            
            // Insert after upload section
            const uploadSection = sidebar.querySelector('.sidebar-upload');
            if (uploadSection) {
                uploadSection.after(panel);
            } else {
                sidebar.appendChild(panel);
            }

            // Add refresh button handler
            const refreshBtn = panel.querySelector('.files-refresh-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    refreshBtn.classList.add('spinning');
                    setTimeout(() => refreshBtn.classList.remove('spinning'), 500);
                    if (window.layoutManager) {
                        this.refresh(
                            window.layoutManager.currentDomain,
                            window.layoutManager.currentDashboard
                        );
                    }
                });
            }
            const defaultBtn = panel.querySelector('.files-default-btn');
            if (defaultBtn) {
                defaultBtn.addEventListener('click', async () => {
                    defaultBtn.classList.add('spinning');
                    await this.resetToDefault(window.layoutManager ? window.layoutManager.currentDomain : 'astronomy');
                    setTimeout(() => defaultBtn.classList.remove('spinning'), 500);
                });
            }
        }
    }

    /**
     * Update the files panel content with enhanced UI
     */
    updatePanel(domain, dashboard) {
        const content = document.getElementById('files-panel-content');
        if (!content) return;

        const key = `${domain}-${dashboard}`;
        const activeFile = this.activeFiles[key];
        const availableFiles = this.availableFiles[key] || [];
        const requiredFeatures = this.getRequiredFeatures(domain, dashboard);

        if (!activeFile && availableFiles.length === 0) {
            content.innerHTML = `
                <div class="files-empty-state">
                    <span class="empty-icon">📁</span>
                    <p>No files available for this dashboard</p>
                </div>
            `;
            return;
        }

        let html = '';

        // Show active file with enhanced card design
        if (activeFile) {
            const fileType = activeFile.type || (activeFile.path && activeFile.path.includes('uploads') ? 'uploaded' : 'raw');
            const coverage = this.evaluateCoverage(activeFile.columns, requiredFeatures);
            html += `
                <div class="active-file-section">
                    <div class="section-label">
                        <span class="label-icon">✓</span>
                        <span>Active File</span>
                    </div>
                    <div class="active-file-card ${fileType}">
                        <div class="file-card-header">
                            <div class="file-icon-wrapper ${fileType}">
                                <span class="file-icon">${fileType === 'uploaded' ? '📤' : '📁'}</span>
                            </div>
                            <div class="file-info">
                                <div class="file-name" title="${activeFile.name}">${activeFile.name}</div>
                                <div class="file-type-badge ${fileType}">${fileType === 'uploaded' ? 'Uploaded' : fileType === 'default' ? 'Default Dataset' : 'Raw Data'}</div>
                            </div>
                        </div>
                        <div class="file-stats-grid">
                            <div class="stat-item">
                                <span class="stat-icon">📊</span>
                                <div class="stat-content">
                                    <span class="stat-value">${activeFile.row_count.toLocaleString()}</span>
                                    <span class="stat-label">Rows</span>
                                </div>
                            </div>
                            <div class="stat-item">
                                <span class="stat-icon">📋</span>
                                <div class="stat-content">
                                    <span class="stat-value">${activeFile.columns.length}</span>
                                    <span class="stat-label">Columns</span>
                                </div>
                            </div>
                            <div class="stat-item">
                                <span class="stat-icon">💾</span>
                                <div class="stat-content">
                                    <span class="stat-value">${this.formatFileSize(activeFile.size)}</span>
                                    <span class="stat-label">Size</span>
                                </div>
                            </div>
                        </div>
                        ${activeFile.columns && activeFile.columns.length > 0 ? `
                            <div class="file-columns-preview">
                                <div class="columns-label">Columns:</div>
                                <div class="columns-tags">
                                    ${activeFile.columns.slice(0, 6).map(col => 
                                        `<span class="column-tag" title="${col}">${this.truncateFileName(col, 12)}</span>`
                                    ).join('')}
                                    ${activeFile.columns.length > 6 ? `<span class="column-tag more">+${activeFile.columns.length - 6} more</span>` : ''}
                                </div>
                            </div>
                        ` : ''}
                        <div class="file-coverage ${coverage.status}">
                            <div class="coverage-pill ${coverage.status}">
                                ${this.getCoverageLabel(coverage.status)}
                            </div>
                            <div class="coverage-detail">
                                ${this.getCoverageSummary(coverage, requiredFeatures)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Show available files with improved list design
        if (availableFiles.length > 0) {
            html += `
                <div class="available-files-section">
                    <div class="section-label">
                        <span class="label-icon">📂</span>
                        <span>Available Files (${availableFiles.length})</span>
                    </div>
                    <div class="available-files-list">
            `;
            
            availableFiles.forEach((file, index) => {
                const isActive = activeFile && file.path === activeFile.path;
                const fileType = file.type || (file.path && file.path.includes('uploads') ? 'uploaded' : 'raw');
                const coverage = this.evaluateCoverage(file.columns, requiredFeatures);
                html += `
                    <div class="available-file-item ${isActive ? 'active' : ''} ${fileType}" 
                         data-filepath="${file.path}"
                         data-dashboard="${dashboard}"
                         data-index="${index}">
                        <div class="file-item-content">
                            <div class="file-item-header">
                                <div class="file-icon-wrapper ${fileType}">
                                    <span class="file-icon">${fileType === 'uploaded' ? '📤' : '📁'}</span>
                                </div>
                                <div class="file-item-info">
                                    <div class="file-name" title="${file.name}">${file.name}</div>
                                    <div class="file-meta">
                                        <span class="meta-item">
                                            <span class="meta-icon">📊</span>
                                            ${file.row_count.toLocaleString()} rows
                                        </span>
                                        <span class="meta-item">
                                            <span class="meta-icon">📋</span>
                                            ${file.columns.length} cols
                                        </span>
                                    </div>
                                </div>
                                <div class="coverage-mini ${coverage.status}" title="${this.getCoverageTooltip(coverage, requiredFeatures)}">
                                    ${this.getCoverageShortLabel(coverage.status)}
                                </div>
                                ${isActive ? `
                                    <div class="active-badge">
                                        <span class="badge-icon">✓</span>
                                        <span>Active</span>
                                    </div>
                                ` : ''}
                            </div>
                            ${requiredFeatures.length > 0 ? `
                                <div class="coverage-summary ${coverage.status}">
                                    ${this.getCoverageSummary(coverage, requiredFeatures)}
                                </div>
                            ` : ''}
                            ${!isActive ? `
                                <button class="use-file-btn" data-filepath="${file.path}" title="Use this file">
                                    <span class="btn-icon">→</span>
                                    <span>Use File</span>
                                </button>
                            ` : ''}
                        </div>
                    </div>
                `;
            });

            html += `
                    </div>
                </div>
            `;
        }

        content.innerHTML = html;

        // Add click handlers for "Use" buttons
        content.querySelectorAll('.use-file-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const filepath = e.target.closest('.use-file-btn').dataset.filepath;
                const dashboard = e.target.closest('.available-file-item').dataset.dashboard;
                
                // Add loading state
                btn.classList.add('loading');
                btn.innerHTML = '<span class="btn-icon spinning">⟳</span><span>Switching...</span>';
                
                await this.setActiveFile(domain, dashboard, filepath);
            });
        });

        // Add hover effects and animations
        content.querySelectorAll('.available-file-item').forEach(item => {
            item.addEventListener('mouseenter', function() {
                if (!this.classList.contains('active')) {
                    this.style.transform = 'translateX(4px)';
                }
            });
            item.addEventListener('mouseleave', function() {
                this.style.transform = '';
            });
        });
    }

    /**
     * Set active file for a dashboard
     */
    async setActiveFile(domain, dashboard, filepath) {
        try {
            const response = await fetch(`/api/${domain}/data/files/set`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dashboard, filepath })
            });

            const data = await response.json();
            if (data.status === 'success') {
                // Refresh file info
                await this.refresh(domain, dashboard);
                this.updatePanel(domain, dashboard);
                
                // Trigger dashboard refresh
                if (window.layoutManager) {
                    if (window.layoutManager.datasetContexts) {
                        delete window.layoutManager.datasetContexts[domain];
                    }
                    window.layoutManager.loadDashboard(domain, dashboard);
                }
            } else {
                throw new Error(data.error || 'Failed to set active file');
            }
        } catch (error) {
            console.error('Failed to set active file:', error);
            alert('Failed to change data file. Please try again.');
        }
    }

    async resetToDefault(domain) {
        try {
            const response = await fetch(`/api/${domain}/data/files/reset`, {
                method: 'POST'
            });
            const data = await response.json();
            if (data.status === 'success') {
                if (window.layoutManager) {
                    const dashboard = window.layoutManager.currentDashboard;
                    await this.refresh(domain, dashboard);
                    this.updatePanel(domain, dashboard);
                    if (window.layoutManager.datasetContexts) {
                        delete window.layoutManager.datasetContexts[domain];
                    }
                    window.layoutManager.loadDashboard(domain, dashboard);
                }
            } else {
                throw new Error(data.error || 'Unable to reset to default dataset');
            }
        } catch (error) {
            console.error('Failed to reset to default dataset:', error);
            alert('Could not switch back to the default dataset. Please try again.');
        }
    }

    /**
     * Truncate file name for display
     */
    truncateFileName(name, maxLength) {
        if (name.length <= maxLength) return name;
        return name.substring(0, maxLength - 3) + '...';
    }

    /**
     * Format file size
     */
    formatFileSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    }

    getRequiredFeatures(domain, dashboard) {
        const configs = domain === 'astronomy' ? window.AstronomyDashboards : window.FinanceDashboards;
        if (!configs || !configs[dashboard]) return [];
        const required = configs[dashboard].requiredFeatures;
        return Array.isArray(required) ? required : [];
    }

    evaluateCoverage(columns = [], requiredFeatures = []) {
        if (!requiredFeatures || requiredFeatures.length === 0) {
            return { status: 'neutral', missing: [], coverage: 1 };
        }
        if (!columns || columns.length === 0) {
            return { status: 'missing', missing: requiredFeatures.slice(), coverage: 0 };
        }
        const normalizedColumns = columns.map(col => col.toLowerCase());
        const missing = requiredFeatures.filter(req => !normalizedColumns.includes(req.toLowerCase()));
        let status = 'full';
        if (missing.length === 0) {
            status = 'full';
        } else if (missing.length === requiredFeatures.length) {
            status = 'missing';
        } else {
            status = 'partial';
        }
        const coverage = (requiredFeatures.length - missing.length) / requiredFeatures.length;
        return { status, missing, coverage };
    }

    getCoverageLabel(status) {
        switch (status) {
            case 'full':
                return 'Full coverage';
            case 'partial':
                return 'Partial coverage';
            case 'missing':
                return 'Missing data';
            default:
                return 'No requirements';
        }
    }

    getCoverageShortLabel(status) {
        switch (status) {
            case 'full':
                return 'Full';
            case 'partial':
                return 'Partial';
            case 'missing':
                return 'Missing';
            default:
                return 'N/A';
        }
    }

    getCoverageSummary(coverage, requiredFeatures) {
        if (!requiredFeatures || requiredFeatures.length === 0) {
            return 'No required columns for this dashboard';
        }
        if (coverage.status === 'full') {
            return 'All required columns detected';
        }
        if (coverage.status === 'missing') {
            return 'Missing all required columns';
        }
        return `Missing: ${this.formatMissingColumns(coverage.missing)}`;
    }

    formatMissingColumns(missing = [], limit = 3) {
        if (!missing || missing.length === 0) return '';
        const display = missing.slice(0, limit).join(', ');
        if (missing.length > limit) {
            return `${display} (+${missing.length - limit} more)`;
        }
        return display;
    }

    getCoverageTooltip(coverage, requiredFeatures) {
        if (!requiredFeatures || requiredFeatures.length === 0) {
            return 'This dashboard does not enforce a required schema.';
        }
        if (coverage.status === 'full') {
            return 'All required columns are present in this file.';
        }
        if (coverage.status === 'missing') {
            return `Missing columns: ${coverage.missing.join(', ') || 'all required columns'}`;
        }
        return `Missing columns: ${coverage.missing.join(', ')}`;
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.dashboardFilesPanel = new DashboardFilesPanel();
        window.dashboardFilesPanel.init();
    });
} else {
    window.dashboardFilesPanel = new DashboardFilesPanel();
    window.dashboardFilesPanel.init();
}
