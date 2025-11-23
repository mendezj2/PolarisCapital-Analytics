/**
 * Graph Enhancer
 * Applies consistent formatting, annotations, and empty states to ECharts options.
 */
class GraphEnhancer {
    static enhanceOption(type, option = {}, meta = {}) {
        const enhanced = { ...option };
        const palette = [
            '#6ee7ff', '#8b5cf6', '#22d3ee', '#f9a8d4', '#38bdf8',
            '#c084fc', '#f97316', '#10b981', '#f43f5e', '#a3e635'
        ];

        enhanced.color = enhanced.color || palette;
        enhanced.grid = enhanced.grid || { left: '8%', right: '6%', top: '14%', bottom: '14%' };
        enhanced.textStyle = enhanced.textStyle || { color: 'var(--text-secondary)', fontFamily: 'Inter, system-ui' };

        // Axes formatting with units
        if (type !== 'pie') {
            const xAxis = Array.isArray(enhanced.xAxis) ? enhanced.xAxis[0] : enhanced.xAxis || {};
            const yAxis = Array.isArray(enhanced.yAxis) ? enhanced.yAxis[0] : enhanced.yAxis || {};
            if (meta.xUnit && !xAxis.name) xAxis.name = meta.xUnit;
            if (meta.yUnit && !yAxis.name) yAxis.name = meta.yUnit;
            xAxis.axisLabel = xAxis.axisLabel || {};
            yAxis.axisLabel = yAxis.axisLabel || {};
            xAxis.axisLabel.color = xAxis.axisLabel.color || 'var(--text-muted)';
            yAxis.axisLabel.color = yAxis.axisLabel.color || 'var(--text-muted)';
            xAxis.splitLine = xAxis.splitLine || { show: true, lineStyle: { color: 'rgba(255,255,255,0.04)' } };
            yAxis.splitLine = yAxis.splitLine || { show: true, lineStyle: { color: 'rgba(255,255,255,0.06)' } };
            enhanced.xAxis = Array.isArray(enhanced.xAxis) ? enhanced.xAxis : xAxis;
            enhanced.yAxis = Array.isArray(enhanced.yAxis) ? enhanced.yAxis : yAxis;
        }

        // Tooltip with quick significance text
        enhanced.tooltip = enhanced.tooltip || {};
        enhanced.tooltip.trigger = enhanced.tooltip.trigger || (type === 'pie' ? 'item' : 'axis');
        enhanced.tooltip.backgroundColor = enhanced.tooltip.backgroundColor || 'rgba(15,23,42,0.92)';
        enhanced.tooltip.borderColor = enhanced.tooltip.borderColor || 'var(--border-color)';
        enhanced.tooltip.textStyle = enhanced.tooltip.textStyle || { color: '#e2e8f0' };
        if (!enhanced.tooltip.formatter) {
            enhanced.tooltip.formatter = (params) => {
                const p = Array.isArray(params) ? params[0] : params;
                const label = p?.axisValueLabel || p?.name || meta.title || 'Value';
                const val = p?.data?.value ?? p?.data ?? p?.value ?? '--';
                let units = meta.yUnit ? ` ${meta.yUnit}` : '';
                if (typeof val === 'number') return `${label}: ${val}${units}`;
                if (Array.isArray(val)) return `${label}: ${val.join(', ')}${units}`;
                return `${label}: ${val}`;
            };
        }

        // Highlight bands / confidence if provided
        if (meta?.bands && Array.isArray(meta.bands)) {
            enhanced.visualMap = enhanced.visualMap || {
                show: false,
                pieces: meta.bands.map(b => ({ gt: b.min, lte: b.max, color: b.color || '#22d3ee55' }))
            };
        }

        // Regression / reference lines if stats available
        if (meta?.stats?.r2 && type !== 'pie') {
            enhanced.legend = enhanced.legend || { textStyle: { color: 'var(--text-muted)' } };
            const ref = {
                name: 'Fit',
                type: 'line',
                symbol: 'none',
                lineStyle: { type: 'dashed', opacity: 0.4 },
                data: meta.stats.referenceLine || []
            };
            if (Array.isArray(enhanced.series)) {
                enhanced.series = [...enhanced.series, ref];
            }
        }

        // Empty state guard
        if (GraphEnhancer._isEmpty(enhanced)) {
            return GraphEnhancer._emptyOption(meta.title || 'Insufficient Data', meta.missingFields);
        }

        return enhanced;
    }

    static _isEmpty(option) {
        const series = option.series || [];
        if (!Array.isArray(series) || series.length === 0) return true;
        return series.every(s => !s.data || s.data.length === 0);
    }

    static _emptyOption(title, missingFields = []) {
        return {
            title: {
                text: title,
                left: 'center',
                top: '40%',
                textStyle: { color: 'var(--text-secondary)' },
            },
            graphic: {
                type: 'text',
                left: 'center',
                top: '50%',
                style: {
                    text: missingFields && missingFields.length
                        ? `Missing: ${missingFields.join(', ')}`
                        : 'Upload a CSV with matching columns to populate this chart.',
                    fill: 'var(--text-muted)',
                    fontSize: 13,
                    lineHeight: 20,
                    width: 260,
                }
            }
        };
    }
}

window.GraphEnhancer = GraphEnhancer;
