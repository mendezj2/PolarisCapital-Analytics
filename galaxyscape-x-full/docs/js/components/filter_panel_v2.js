/**
 * FilterPanelV2
 * Compact ribbon + slide-out filter experience that does not push charts downward.
 */
class FilterPanelV2 {
    constructor({ ribbonId, drawerId, filters = [], onChange }) {
        this.ribbonId = ribbonId;
        this.drawerId = drawerId;
        this.filters = filters;
        this.onChange = onChange;
        this.mode = 'mini'; // mini | expanded | hidden
        this.current = {};
    }

    init() {
        this.renderRibbon();
        this.renderDrawer();
    }

    updateFilters(filters = []) {
        this.filters = filters;
        this.renderRibbon();
        this.renderDrawer();
    }

    setMode(mode) {
        this.mode = mode;
        this._syncVisibility();
    }

    cycleMode() {
        const order = ['mini', 'expanded', 'hidden'];
        const idx = order.indexOf(this.mode);
        this.setMode(order[(idx + 1) % order.length]);
        this.renderRibbon();
        this.renderDrawer();
    }

    renderRibbon() {
        const ribbon = document.getElementById(this.ribbonId);
        if (!ribbon) return;
        ribbon.innerHTML = '';
        ribbon.className = 'filter-ribbon';
        if (!this.filters || this.filters.length === 0) {
            ribbon.style.display = 'none';
            return;
        }
        ribbon.style.display = 'flex';

        const summary = document.createElement('div');
        summary.className = 'filter-summary';
        const miniFilters = this.filters.slice(0, 4);
        miniFilters.forEach(f => {
            const chip = this._createChip(f);
            summary.appendChild(chip);
        });

        const actions = document.createElement('div');
        actions.className = 'filter-actions';
        const toggle = document.createElement('button');
        toggle.className = 'filter-toggle-btn';
        toggle.textContent = this.mode === 'expanded' ? 'Hide Filters' : 'More Filters';
        toggle.addEventListener('click', () => {
            this.setMode(this.mode === 'expanded' ? 'mini' : 'expanded');
            toggle.textContent = this.mode === 'expanded' ? 'Hide Filters' : 'More Filters';
        });
        actions.appendChild(toggle);

        ribbon.appendChild(summary);
        ribbon.appendChild(actions);
        this._syncVisibility();
    }

    renderDrawer() {
        const drawer = document.getElementById(this.drawerId);
        if (!drawer) return;
        drawer.innerHTML = '';
        drawer.className = 'filter-drawer';
        if (!this.filters || this.filters.length === 0) {
            drawer.style.display = 'none';
            return;
        }

        const header = document.createElement('div');
        header.className = 'filter-drawer__header';
        header.innerHTML = '<h4>Filters</h4>';
        const closeBtn = document.createElement('button');
        closeBtn.className = 'filter-toggle-btn';
        closeBtn.textContent = 'Close';
        closeBtn.addEventListener('click', () => this.setMode('mini'));
        header.appendChild(closeBtn);
        drawer.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'filter-grid';
        this.filters.forEach(filter => {
            const control = this._createControl(filter);
            grid.appendChild(control);
        });
        drawer.appendChild(grid);
        this._syncVisibility();
    }

    _syncVisibility() {
        const drawer = document.getElementById(this.drawerId);
        const ribbon = document.getElementById(this.ribbonId);
        if (!drawer || !ribbon) return;
        drawer.style.display = this.mode === 'expanded' ? 'block' : 'none';
        ribbon.style.opacity = this.mode === 'hidden' ? '0.25' : '1';
        ribbon.style.pointerEvents = this.mode === 'hidden' ? 'none' : 'auto';
    }

    _createChip(filter) {
        const chip = document.createElement('div');
        chip.className = 'filter-chip';
        const label = document.createElement('span');
        label.textContent = filter.label || filter.name;
        chip.appendChild(label);

        const valueSlot = document.createElement('div');
        valueSlot.className = 'filter-chip__input';
        const control = this._createInput(filter, true);
        valueSlot.appendChild(control);
        chip.appendChild(valueSlot);
        return chip;
    }

    _createControl(filter) {
        const wrapper = document.createElement('div');
        wrapper.className = 'filter-tile';
        const label = document.createElement('label');
        label.textContent = filter.label || filter.name;
        wrapper.appendChild(label);
        wrapper.appendChild(this._createInput(filter, false));
        return wrapper;
    }

    _createInput(filter, compact = false) {
        const onChange = (value) => {
            this.current[filter.name] = value;
            this.onChange && this.onChange(this.current);
        };

        switch (filter.type) {
            case 'search': {
                const input = document.createElement('input');
                input.type = 'search';
                input.placeholder = filter.placeholder || 'Search';
                input.value = '';
                input.addEventListener('input', (e) => {
                    if (filter.debounce) {
                        clearTimeout(input._debounce);
                        input._debounce = setTimeout(() => onChange(e.target.value), filter.debounce);
                    } else {
                        onChange(e.target.value);
                    }
                });
                return input;
            }
            case 'dropdown':
            case 'single-select': {
                const select = document.createElement('select');
                (filter.options || []).forEach(opt => {
                    const o = document.createElement('option');
                    o.value = opt.value;
                    o.textContent = opt.label;
                    select.appendChild(o);
                });
                select.value = filter.defaultValue || (filter.options?.[0]?.value || '');
                onChange(select.value);
                select.addEventListener('change', (e) => onChange(e.target.value));
                return select;
            }
            case 'multi-select': {
                const select = document.createElement('select');
                select.multiple = !compact;
                (filter.options || []).forEach(opt => {
                    const o = document.createElement('option');
                    o.value = opt.value;
                    o.textContent = opt.label;
                    select.appendChild(o);
                });
                select.addEventListener('change', () => {
                    const values = Array.from(select.selectedOptions).map(o => o.value);
                    onChange(values);
                });
                return select;
            }
            case 'slider': {
                const wrapper = document.createElement('div');
                wrapper.className = 'filter-slider';
                const range = document.createElement('input');
                range.type = 'range';
                range.min = filter.min ?? 0;
                range.max = filter.max ?? 100;
                range.step = filter.step ?? 1;
                range.value = filter.defaultValue ?? filter.min ?? 0;
                const value = document.createElement('span');
                value.textContent = range.value;
                range.addEventListener('input', () => {
                    value.textContent = range.value;
                    onChange(parseFloat(range.value));
                });
                wrapper.appendChild(range);
                wrapper.appendChild(value);
                onChange(parseFloat(range.value));
                return wrapper;
            }
            case 'toggle': {
                const toggle = document.createElement('button');
                toggle.className = 'filter-toggle';
                const onLabel = filter.onLabel || 'On';
                const offLabel = filter.offLabel || 'Off';
                let state = !!filter.defaultValue;
                const sync = () => {
                    toggle.textContent = state ? onLabel : offLabel;
                    toggle.classList.toggle('active', state);
                };
                sync();
                onChange(state);
                toggle.addEventListener('click', () => {
                    state = !state;
                    sync();
                    onChange(state);
                });
                return toggle;
            }
            case 'date-range': {
                const box = document.createElement('div');
                box.className = 'filter-date-range';
                const start = document.createElement('input');
                start.type = 'date';
                const end = document.createElement('input');
                end.type = 'date';
                const update = () => onChange({ start: start.value, end: end.value });
                start.addEventListener('change', update);
                end.addEventListener('change', update);
                box.appendChild(start);
                box.appendChild(end);
                return box;
            }
            case 'number': {
                const number = document.createElement('input');
                number.type = 'number';
                number.placeholder = compact ? '#' : (filter.placeholder || '');
                number.addEventListener('change', (e) => onChange(parseFloat(e.target.value)));
                return number;
            }
            case 'checkbox': {
                const wrap = document.createElement('div');
                wrap.className = 'filter-checkboxes';
                (filter.options || []).forEach(opt => {
                    const label = document.createElement('label');
                    label.style.display = 'flex';
                    label.style.alignItems = 'center';
                    label.style.gap = '0.35rem';
                    const box = document.createElement('input');
                    box.type = 'checkbox';
                    box.value = opt.value;
                    box.addEventListener('change', () => {
                        const values = Array.from(wrap.querySelectorAll('input:checked')).map(i => i.value);
                        onChange(values);
                    });
                    label.appendChild(box);
                    label.appendChild(document.createTextNode(opt.label));
                    wrap.appendChild(label);
                });
                return wrap;
            }
            default: {
                const input = document.createElement('input');
                input.type = 'text';
                input.placeholder = filter.placeholder || filter.label;
                input.addEventListener('input', (e) => onChange(e.target.value));
                return input;
            }
        }
    }
}

window.FilterPanelV2 = FilterPanelV2;
