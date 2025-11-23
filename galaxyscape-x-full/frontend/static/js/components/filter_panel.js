/**
 * Filter Panel Component
 * Provides interactive filters for dashboards (date range, sliders, dropdowns)
 */

class FilterPanel {
    constructor(containerId, config = {}) {
        this.containerId = containerId;
        this.config = {
            filters: config.filters || [],
            onFilterChange: config.onFilterChange || null,
            ...config
        };
        this.currentFilters = {};
    }

    /**
     * Initialize the filter panel
     */
    init() {
        const container = document.getElementById(this.containerId);
        if (!container) {
            console.warn(`Filter panel container ${this.containerId} not found`);
            return;
        }

        this.render();
    }

    /**
     * Render filter controls
     */
    render() {
        const container = document.getElementById(this.containerId);
        if (!container) return;

        container.innerHTML = '';

        this.config.filters.forEach(filter => {
            const filterEl = this._createFilterElement(filter);
            container.appendChild(filterEl);
        });
    }

    /**
     * Create a filter element based on type
     */
    _createFilterElement(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'filter-item';
        wrapper.style.marginBottom = '0';
        wrapper.style.minWidth = '0'; // Prevent overflow

        const label = document.createElement('label');
        label.textContent = filter.label || filter.name;
        label.style.display = 'block';
        label.style.marginBottom = '0.5rem';
        label.style.fontWeight = '600';
        label.style.fontSize = '0.875rem';
        label.style.color = 'var(--text-secondary)';
        wrapper.appendChild(label);

        let input;

        switch (filter.type) {
            case 'date-range':
                input = this._createDateRangeFilter(filter);
                break;
            case 'slider':
                input = this._createSliderFilter(filter);
                break;
            case 'dropdown':
                input = this._createDropdownFilter(filter);
                break;
            case 'multi-select':
                input = this._createMultiSelectFilter(filter);
                break;
            case 'number':
                input = this._createNumberFilter(filter);
                break;
            case 'checkbox':
                input = this._createCheckboxFilter(filter);
                break;
            case 'toggle':
                input = this._createToggleFilter(filter);
                break;
            case 'search':
                input = this._createSearchFilter(filter);
                break;
            default:
                input = document.createElement('input');
                input.type = 'text';
        }

        wrapper.appendChild(input);
        return wrapper;
    }

    /**
     * Create date range filter
     */
    _createDateRangeFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'date-range-wrapper';
        wrapper.style.display = 'flex';
        wrapper.style.gap = '0.5rem';
        wrapper.style.alignItems = 'center';
        wrapper.style.width = '100%';
        wrapper.style.minWidth = '0';
        wrapper.style.boxSizing = 'border-box';

        const startInput = document.createElement('input');
        startInput.type = 'date';
        startInput.id = `${filter.name}_start`;
        startInput.value = filter.defaultStart || '';
        startInput.style.flex = '1';
        startInput.style.padding = '0.5rem';
        startInput.style.border = '1px solid var(--border-color)';
        startInput.style.borderRadius = '4px';
        startInput.style.background = 'var(--bg-secondary)';
        startInput.style.color = 'var(--text-primary)';

        const endInput = document.createElement('input');
        endInput.type = 'date';
        endInput.id = `${filter.name}_end`;
        endInput.value = filter.defaultEnd || '';
        endInput.style.flex = '1';
        endInput.style.padding = '0.5rem';
        endInput.style.border = '1px solid var(--border-color)';
        endInput.style.borderRadius = '4px';
        endInput.style.background = 'var(--bg-secondary)';
        endInput.style.color = 'var(--text-primary)';

        const updateFilter = () => {
            // Store both as date_range object and individual fields
            this.currentFilters[filter.name] = {
                start: startInput.value,
                end: endInput.value
            };
            // Also store as individual fields for backend compatibility
            if (startInput.value) this.currentFilters.date_start = startInput.value;
            if (endInput.value) this.currentFilters.date_end = endInput.value;
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        };

        startInput.addEventListener('change', updateFilter);
        endInput.addEventListener('change', updateFilter);

        wrapper.appendChild(startInput);
        wrapper.appendChild(endInput);

        return wrapper;
    }

    /**
     * Create slider filter
     */
    _createSliderFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'slider-wrapper';
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.gap = '0.5rem';
        wrapper.style.width = '100%';
        wrapper.style.minWidth = '0';
        wrapper.style.boxSizing = 'border-box';

        const slider = document.createElement('input');
        slider.type = 'range';
        slider.id = filter.name;
        slider.min = filter.min || 0;
        slider.max = filter.max || 100;
        slider.value = filter.defaultValue || filter.min || 0;
        slider.step = filter.step || 1;
        slider.style.width = '100%';
        slider.style.minWidth = '0';
        slider.style.boxSizing = 'border-box';

        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = slider.value;
        valueDisplay.style.marginLeft = '0';
        valueDisplay.style.marginTop = '0.25rem';
        valueDisplay.style.color = 'var(--accent-primary)';
        valueDisplay.style.fontWeight = '600';
        valueDisplay.style.fontSize = '0.875rem';
        valueDisplay.style.textAlign = 'center';

        slider.addEventListener('input', () => {
            valueDisplay.textContent = slider.value;
            this.currentFilters[filter.name] = parseFloat(slider.value);
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        });

        wrapper.appendChild(slider);
        wrapper.appendChild(valueDisplay);

        return wrapper;
    }

    /**
     * Create dropdown filter
     */
    _createDropdownFilter(filter) {
        const select = document.createElement('select');
        select.id = filter.name;
        select.style.width = '100%';
        select.style.padding = '0.5rem';
        select.style.border = '1px solid var(--border-color)';
        select.style.borderRadius = '4px';
        select.style.background = 'var(--bg-secondary)';
        select.style.color = 'var(--text-primary)';

        if (filter.options) {
            filter.options.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option.value;
                optionEl.textContent = option.label || option.value;
                if (option.value === filter.defaultValue) {
                    optionEl.selected = true;
                }
                select.appendChild(optionEl);
            });
        }

        select.addEventListener('change', () => {
            this.currentFilters[filter.name] = select.value;
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        });

        return select;
    }

    /**
     * Create number input filter
     */
    _createNumberFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.gap = '0.5rem';

        const minInput = document.createElement('input');
        minInput.type = 'number';
        minInput.id = `${filter.name}_min`;
        minInput.placeholder = 'Min';
        minInput.value = filter.defaultMin || '';
        minInput.style.flex = '1';
        minInput.style.minWidth = '0';
        minInput.style.padding = '0.5rem';
        minInput.style.border = '1px solid var(--border-color)';
        minInput.style.borderRadius = '4px';
        minInput.style.background = 'var(--bg-secondary)';
        minInput.style.color = 'var(--text-primary)';
        minInput.style.boxSizing = 'border-box';

        const maxInput = document.createElement('input');
        maxInput.type = 'number';
        maxInput.id = `${filter.name}_max`;
        maxInput.placeholder = 'Max';
        maxInput.value = filter.defaultMax || '';
        maxInput.style.flex = '1';
        maxInput.style.minWidth = '0';
        maxInput.style.padding = '0.5rem';
        maxInput.style.border = '1px solid var(--border-color)';
        maxInput.style.borderRadius = '4px';
        maxInput.style.background = 'var(--bg-secondary)';
        maxInput.style.color = 'var(--text-primary)';
        maxInput.style.boxSizing = 'border-box';

        const updateFilter = () => {
            this.currentFilters[filter.name] = {
                min: minInput.value ? parseFloat(minInput.value) : null,
                max: maxInput.value ? parseFloat(maxInput.value) : null
            };
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        };

        minInput.addEventListener('change', updateFilter);
        maxInput.addEventListener('change', updateFilter);

        wrapper.appendChild(minInput);
        wrapper.appendChild(maxInput);

        return wrapper;
    }

    /**
     * Get current filter values
     */
    getFilters() {
        return this.currentFilters;
    }

    /**
     * Create multi-select dropdown filter
     */
    _createMultiSelectFilter(filter) {
        const select = document.createElement('select');
        select.id = filter.name;
        select.multiple = true;
        select.style.width = '100%';
        select.style.padding = '0.5rem';
        select.style.border = '1px solid var(--border-color)';
        select.style.borderRadius = '4px';
        select.style.background = 'var(--bg-secondary)';
        select.style.color = 'var(--text-primary)';
        select.style.minHeight = '100px';

        if (filter.options) {
            filter.options.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option.value;
                optionEl.textContent = option.label || option.value;
                if (filter.defaultValue && filter.defaultValue.includes(option.value)) {
                    optionEl.selected = true;
                }
                select.appendChild(optionEl);
            });
        }

        select.addEventListener('change', () => {
            const selected = Array.from(select.selectedOptions).map(opt => opt.value);
            this.currentFilters[filter.name] = selected;
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        });

        // Initialize with default values
        if (filter.defaultValue) {
            this.currentFilters[filter.name] = filter.defaultValue;
        }

        return select;
    }

    /**
     * Create checkbox filter
     */
    _createCheckboxFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.style.display = 'flex';
        wrapper.style.flexDirection = 'column';
        wrapper.style.gap = '0.5rem';

        if (filter.options) {
            filter.options.forEach(option => {
                const checkboxWrapper = document.createElement('div');
                checkboxWrapper.style.display = 'flex';
                checkboxWrapper.style.alignItems = 'center';
                checkboxWrapper.style.gap = '0.5rem';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.id = `${filter.name}_${option.value}`;
                checkbox.value = option.value;
                checkbox.checked = filter.defaultValue && filter.defaultValue.includes(option.value);

                const checkboxLabel = document.createElement('label');
                checkboxLabel.htmlFor = checkbox.id;
                checkboxLabel.textContent = option.label || option.value;
                checkboxLabel.style.fontWeight = 'normal';
                checkboxLabel.style.cursor = 'pointer';

                checkbox.addEventListener('change', () => {
                    const checked = Array.from(wrapper.querySelectorAll('input[type="checkbox"]:checked'))
                        .map(cb => cb.value);
                    this.currentFilters[filter.name] = checked;
                    if (this.config.onFilterChange) {
                        this.config.onFilterChange(this.currentFilters);
                    }
                });

                checkboxWrapper.appendChild(checkbox);
                checkboxWrapper.appendChild(checkboxLabel);
                wrapper.appendChild(checkboxWrapper);
            });
        }

        // Initialize with default values
        if (filter.defaultValue) {
            this.currentFilters[filter.name] = filter.defaultValue;
        }

        return wrapper;
    }

    /**
     * Create toggle switch filter
     */
    _createToggleFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'toggle-wrapper';
        wrapper.style.display = 'flex';
        wrapper.style.alignItems = 'center';
        wrapper.style.gap = '0.75rem';
        wrapper.style.width = '100%';
        wrapper.style.boxSizing = 'border-box';

        const toggle = document.createElement('label');
        toggle.style.position = 'relative';
        toggle.style.display = 'inline-block';
        toggle.style.width = '50px';
        toggle.style.height = '24px';

        const input = document.createElement('input');
        input.type = 'checkbox';
        input.id = filter.name;
        input.checked = filter.defaultValue || false;
        input.style.opacity = '0';
        input.style.width = '0';
        input.style.height = '0';

        const slider = document.createElement('span');
        slider.style.position = 'absolute';
        slider.style.cursor = 'pointer';
        slider.style.top = '0';
        slider.style.left = '0';
        slider.style.right = '0';
        slider.style.bottom = '0';
        slider.style.backgroundColor = input.checked ? '#22c55e' : '#ccc';
        slider.style.transition = '0.4s';
        slider.style.borderRadius = '24px';

        const sliderBefore = document.createElement('span');
        sliderBefore.style.position = 'absolute';
        sliderBefore.style.content = '""';
        sliderBefore.style.height = '18px';
        sliderBefore.style.width = '18px';
        sliderBefore.style.left = input.checked ? '26px' : '3px';
        sliderBefore.style.bottom = '3px';
        sliderBefore.style.backgroundColor = 'white';
        sliderBefore.style.transition = '0.4s';
        sliderBefore.style.borderRadius = '50%';

        input.addEventListener('change', () => {
            slider.style.backgroundColor = input.checked ? '#22c55e' : '#ccc';
            sliderBefore.style.left = input.checked ? '26px' : '3px';
            this.currentFilters[filter.name] = input.checked;
            if (this.config.onFilterChange) {
                this.config.onFilterChange(this.currentFilters);
            }
        });

        toggle.appendChild(input);
        toggle.appendChild(slider);
        slider.appendChild(sliderBefore);

        const labelText = document.createElement('span');
        labelText.textContent = filter.labelText || (input.checked ? filter.onLabel || 'On' : filter.offLabel || 'Off');
        labelText.style.fontSize = '0.875rem';
        labelText.style.color = 'var(--text-secondary)';

        input.addEventListener('change', () => {
            labelText.textContent = input.checked ? (filter.onLabel || 'On') : (filter.offLabel || 'Off');
        });

        wrapper.appendChild(toggle);
        wrapper.appendChild(labelText);

        // Initialize
        this.currentFilters[filter.name] = input.checked;

        return wrapper;
    }

    /**
     * Create search input filter
     */
    _createSearchFilter(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'search-wrapper';
        wrapper.style.position = 'relative';
        wrapper.style.width = '100%';
        wrapper.style.boxSizing = 'border-box';

        const input = document.createElement('input');
        input.type = 'text';
        input.id = filter.name;
        input.placeholder = filter.placeholder || 'Search...';
        input.value = filter.defaultValue || '';
        input.style.width = '100%';
        input.style.padding = '0.5rem 2.5rem 0.5rem 0.75rem';
        input.style.border = '1px solid var(--border-color)';
        input.style.borderRadius = '4px';
        input.style.background = 'var(--bg-secondary)';
        input.style.color = 'var(--text-primary)';

        const searchIcon = document.createElement('span');
        searchIcon.textContent = '🔍';
        searchIcon.style.position = 'absolute';
        searchIcon.style.right = '0.75rem';
        searchIcon.style.top = '50%';
        searchIcon.style.transform = 'translateY(-50%)';
        searchIcon.style.pointerEvents = 'none';

        let debounceTimer;
        input.addEventListener('input', (e) => {
            e.stopPropagation();
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => {
                const value = input.value.trim();
                this.currentFilters[filter.name] = value;
                if (this.config.onFilterChange) {
                    this.config.onFilterChange(this.currentFilters);
                }
            }, filter.debounce || 300);
        });
        
        // Also trigger on Enter key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                e.stopPropagation();
                clearTimeout(debounceTimer);
                const value = input.value.trim();
                this.currentFilters[filter.name] = value;
                if (this.config.onFilterChange) {
                    this.config.onFilterChange(this.currentFilters);
                }
            }
        });

        wrapper.appendChild(input);
        wrapper.appendChild(searchIcon);

        // Initialize
        if (input.value) {
            this.currentFilters[filter.name] = input.value;
        }

        return wrapper;
    }

    /**
     * Reset all filters
     */
    reset() {
        this.currentFilters = {};
        this.render();
    }

    /**
     * Set filter values programmatically
     */
    setFilters(filters) {
        this.currentFilters = { ...this.currentFilters, ...filters };
        this.render();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = FilterPanel;
}
