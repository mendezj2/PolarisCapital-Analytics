/**
 * Main Application Controller
 * Handles file uploads, mode switching, and coordinates with layout manager
 */

const state = {
    mode: 'astronomy',
    datasetId: null,
    currentData: null,
    currentDashboard: 'overview'
};

const API_BASE = (window.GALAXYSCAPE_API_BASE || '').replace(/\/$/, '');
const nativeFetch = window.fetch.bind(window);
window.fetch = (resource, init) => {
    if (typeof resource === 'string' && resource.startsWith('/api/')) {
        const target = API_BASE ? `${API_BASE}${resource}` : resource;
        return nativeFetch(target, init);
    }
    return nativeFetch(resource, init);
};

window.defaultDatasetSummary = window.defaultDatasetSummary || {
    astronomy: null,
    finance: null
};

// Toast notification system
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Loading overlay
function showLoading() {
    let overlay = document.getElementById('loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'loading-overlay';
        overlay.innerHTML = '<div class="loading-spinner"></div>';
        document.body.appendChild(overlay);
    }
    overlay.classList.add('active');
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('active');
    }
}

async function refreshDataFiles() {
    try {
        const domain = state.mode;
        const resp = await fetch(`/api/${domain}/data/files`);
        const data = await resp.json();
        const listEl = document.getElementById('data-files-list');
        if (!listEl) return;
        listEl.innerHTML = '';
        if (!data.files || data.files.length === 0) {
            listEl.innerHTML = '<li class="data-file-item muted">No files found</li>';
            return;
        }
        data.files.slice(0, 8).forEach(file => {
            const li = document.createElement('li');
            li.className = 'data-file-item';
            const name = file.name || (file.path ? file.path.split('/').pop() : 'Unknown');
            const cols = (file.columns || []).slice(0, 4).join(', ');
            const badge = file.rows && file.columns && file.columns.length > 0 ? 'ready' : 'partial';
            li.innerHTML = `<span class="file-name">${name}</span><span class="file-meta">${file.rows || 0} rows · ${cols || 'cols'}</span><span class="file-badge">${badge}</span>`;
            listEl.appendChild(li);
        });
    } catch (err) {
        console.error('Failed to load data files', err);
    }
}
window.refreshDataFiles = refreshDataFiles;

/**
 * Initialize main application
 */
async function fetchDefaultSummary(domain) {
    try {
        const resp = await fetch(`/api/${domain}/default`);
        if (resp.ok) {
            const data = await resp.json();
            window.defaultDatasetSummary[domain] = data;
        }
    } catch (err) {
        console.warn(`Failed to preload ${domain} default summary`, err);
    }
}

function preloadDefaultSummaries() {
    fetchDefaultSummary('astronomy');
    fetchDefaultSummary('finance');
}

function init() {
    preloadDefaultSummaries();
    setupUploadHandler();
    refreshDataFiles();
    // Layout manager handles mode switching and dashboard navigation
    // TODO (USER): Add initialization of other app-wide features
}

/**
 * Setup file upload handler
 */
function setupUploadHandler() {
    const uploadForm = document.getElementById('upload-form');
    const uploadTrigger = document.getElementById('upload-trigger');
    const fileInput = document.getElementById('file-input');
    const uploadStatus = document.getElementById('upload-status');

    if (!uploadForm || !uploadTrigger || !fileInput) return;

    // Trigger file input when button is clicked
    uploadTrigger.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const names = Array.from(e.target.files).map(f => f.name);
            const label = names.length === 1 ? names[0] : `${names.length} files selected`;
            uploadTrigger.innerHTML = `<img src="https://cdn-icons-png.flaticon.com/512/1828/1828778.png" alt="Upload" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 6px;"> ${label.length > 24 ? label.substring(0, 24) + '...' : label}`;
            uploadForm.querySelector('.upload-submit').style.display = 'block';
        }
    });

    // Handle form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        if (!fileInput.files.length) {
            uploadStatus.textContent = 'Please select a file';
            return;
        }

        const formData = new FormData();
        Array.from(fileInput.files).forEach(f => formData.append('file', f));

        const endpoint = state.mode === 'astronomy' 
            ? '/api/astronomy/upload' 
            : '/api/finance/upload';
        
        uploadStatus.textContent = 'Uploading...';
        uploadStatus.className = 'upload-status';
        showLoading();

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            hideLoading();
            
            if (data.status === 'success') {
                uploadStatus.textContent = `✅ Uploaded: ${data.row_count} rows`;
                uploadStatus.className = 'upload-status success';
                state.currentData = data;
                
                showToast(`Successfully uploaded ${data.row_count} rows`, 'success');
                // Learning note: uploads land in uploads/{domain} and are auto-scanned
                // by the data context so dashboards can pick the best CSV without reloads.
                
                // Trigger dashboard refresh
                if (window.layoutManager) {
                    window.layoutManager.loadDashboard(state.mode, state.currentDashboard);
                }
                refreshDataFiles();
                
                // Reset form
                fileInput.value = '';
                uploadTrigger.innerHTML = '<img src="https://cdn-icons-png.flaticon.com/512/1828/1828778.png" alt="Upload" style="width: 16px; height: 16px; vertical-align: middle; margin-right: 6px;"> Upload CSV';
                uploadForm.querySelector('.upload-submit').style.display = 'none';
            } else {
                uploadStatus.textContent = `❌ Error: ${data.error || 'Upload failed'}`;
                uploadStatus.className = 'upload-status error';
                showToast(data.error || 'Upload failed', 'error');
            }
        } catch (error) {
            hideLoading();
            uploadStatus.textContent = `❌ Error: ${error.message}`;
            uploadStatus.className = 'upload-status error';
            showToast(`Upload error: ${error.message}`, 'error');
        }
    });
}

/**
 * Handle domain/mode switching
 * This is now handled by layout_manager.js, but kept for backward compatibility
 */
function switchMode(nextMode) {
    if (!nextMode || nextMode === state.mode) return;
    state.mode = nextMode;
    
    // Layout manager will handle the actual switching
    if (window.layoutManager) {
        window.layoutManager.switchDomain(nextMode);
    }
}

// Download sample data handlers
function setupDownloadButtons() {
    const downloadAstroBtn = document.getElementById('download-astro-btn');
    const downloadFinanceBtn = document.getElementById('download-finance-btn');
    
    if (downloadAstroBtn) {
        downloadAstroBtn.addEventListener('click', async () => {
            showLoading();
            try {
                const response = await fetch('/api/astronomy/data/download', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({output_filename: 'astronomy_sample.csv'})
                });
                const result = await response.json();
                hideLoading();
                
                if (result.status === 'success') {
                    showToast('Astronomy data downloaded successfully!', 'success');
                    if (window.layoutManager) {
                        window.layoutManager.loadDashboard('astronomy', 'overview');
                    }
                } else {
                    showToast(result.error || 'Download failed', 'error');
                }
            } catch (error) {
                hideLoading();
                showToast(`Download error: ${error.message}`, 'error');
            }
        });
    }
    
    if (downloadFinanceBtn) {
        downloadFinanceBtn.addEventListener('click', async () => {
            showLoading();
            try {
                const response = await fetch('/api/finance/data/download', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        output_filename: 'finance_sample.csv',
                        tickers: ['AAPL', 'MSFT', 'GOOGL']
                    })
                });
                const result = await response.json();
                hideLoading();
                
                if (result.status === 'success') {
                    showToast('Finance data downloaded successfully!', 'success');
                    if (window.layoutManager) {
                        window.layoutManager.loadDashboard('finance', 'risk');
                    }
                } else {
                    showToast(result.error || 'Download failed', 'error');
                }
            } catch (error) {
                hideLoading();
                showToast(`Download error: ${error.message}`, 'error');
            }
        });
    }
}

// Setup report generation button
function setupReportButton() {
    const reportBtn = document.getElementById('generate-report-btn');
    if (reportBtn) {
        reportBtn.addEventListener('click', async (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const domain = state.mode;
            if (domain === 'finance') {
                // Add loading state
                const icon = reportBtn.querySelector('img');
                if (icon) {
                    icon.style.animation = 'spin 1s linear infinite';
                }
                reportBtn.disabled = true;
                reportBtn.style.opacity = '0.6';
                reportBtn.style.cursor = 'wait';
                
                try {
                    showLoading();
                    const response = await fetch('/api/finance/report');
                    if (response.ok) {
                        const blob = await response.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = `risk_report_${new Date().toISOString().split('T')[0]}.html`;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                        showToast('Report generated successfully!', 'success');
                    } else {
                        showToast('Failed to generate report', 'error');
                    }
                    hideLoading();
                } catch (error) {
                    hideLoading();
                    showToast(`Report error: ${error.message}`, 'error');
                } finally {
                    // Remove loading state
                    if (icon) {
                        icon.style.animation = '';
                    }
                    reportBtn.disabled = false;
                    reportBtn.style.opacity = '1';
                    reportBtn.style.cursor = 'pointer';
                }
            }
        });
        
        // Make sure icon clicks also trigger the button
        const icon = reportBtn.querySelector('img');
        if (icon) {
            icon.style.pointerEvents = 'none'; // Let clicks pass through to button
        }
    }
}

// Setup refresh button
function setupRefreshButton() {
    const refreshBtn = document.getElementById('refresh-dashboard-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            // Add loading state
            const icon = refreshBtn.querySelector('img');
            if (icon) {
                icon.style.animation = 'spin 1s linear infinite';
            }
            refreshBtn.disabled = true;
            refreshBtn.style.opacity = '0.6';
            refreshBtn.style.cursor = 'wait';
            
            if (window.layoutManager) {
                window.layoutManager.loadDashboard(
                    window.layoutManager.currentDomain,
                    window.layoutManager.currentDashboard
                );
                showToast('Dashboard refreshed', 'success');
            }
            
            // Remove loading state after a delay
            setTimeout(() => {
                if (icon) {
                    icon.style.animation = '';
                }
                refreshBtn.disabled = false;
                refreshBtn.style.opacity = '1';
                refreshBtn.style.cursor = 'pointer';
            }, 1000);
        });
        
        // Make sure icon clicks also trigger the button
        const icon = refreshBtn.querySelector('img');
        if (icon) {
            icon.style.pointerEvents = 'none'; // Let clicks pass through to button
        }
    }
}

// Setup toggle filters button
function setupToggleFiltersButton() {
    const toggleFiltersBtn = document.getElementById('toggle-filters-btn');
    if (toggleFiltersBtn) {
        // Ensure button is clickable (handle icon clicks)
        toggleFiltersBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            const filterPanel = document.getElementById('global-filter-panel');
            if (filterPanel) {
                const isVisible = filterPanel.style.display !== 'none' && filterPanel.style.display !== '';
                filterPanel.style.display = isVisible ? 'none' : 'block';
                toggleFiltersBtn.classList.toggle('active', !isVisible);
                
                // Update icon based on state
                const icon = toggleFiltersBtn.querySelector('img');
                if (icon) {
                    if (!isVisible) {
                        icon.src = 'https://cdn-icons-png.flaticon.com/512/1828/1828841.png'; // Filter icon
                        icon.alt = 'Show Filters';
                    } else {
                        icon.src = 'https://cdn-icons-png.flaticon.com/512/1828/1828842.png'; // Filter off icon
                        icon.alt = 'Hide Filters';
                    }
                }
                
                showToast(isVisible ? 'Filters hidden' : 'Filters shown', 'info');
            } else {
                showToast('No filters available for this dashboard', 'info');
            }
        });
        
        // Make sure icon clicks also trigger the button
        const icon = toggleFiltersBtn.querySelector('img');
        if (icon) {
            icon.style.pointerEvents = 'none'; // Let clicks pass through to button
        }
    }
}

// Setup notifications button
function setupNotificationsButton() {
    const notificationsBtn = document.getElementById('notifications-btn');
    if (notificationsBtn) {
        notificationsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            showToast('Notifications feature coming soon!', 'info');
            // TODO: Implement notifications panel
        });
        
        // Make sure icon clicks also trigger the button
        const icon = notificationsBtn.querySelector('img');
        if (icon) {
            icon.style.pointerEvents = 'none'; // Let clicks pass through to button
        }
    }
}

// Setup settings button
function setupSettingsButton() {
    const settingsBtn = document.getElementById('settings-btn');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            showToast('Settings feature coming soon!', 'info');
            // TODO: Implement settings panel
        });
        
        // Make sure icon clicks also trigger the button
        const icon = settingsBtn.querySelector('img');
        if (icon) {
            icon.style.pointerEvents = 'none'; // Let clicks pass through to button
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        init();
        setupDownloadButtons();
        setupReportButton();
        setupRefreshButton();
        setupToggleFiltersButton();
        setupNotificationsButton();
        setupSettingsButton();
    });
} else {
    init();
    setupDownloadButtons();
    setupReportButton();
    setupRefreshButton();
    setupToggleFiltersButton();
    setupNotificationsButton();
    setupSettingsButton();
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showToast('An unexpected error occurred', 'error');
});

// Add smooth page transitions
document.addEventListener('DOMContentLoaded', () => {
    // Add fade-in animation to body
    document.body.style.opacity = '0';
    setTimeout(() => {
        document.body.style.transition = 'opacity 0.5s ease';
        document.body.style.opacity = '1';
    }, 100);
});
