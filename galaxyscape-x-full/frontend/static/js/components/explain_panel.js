/**
 * Explain Panel
 * Lightweight modal that explains the purpose, data fields, and models behind each chart.
 */
class ExplainPanel {
    constructor() {
        this.active = null;
    }

    show(explanation = {}) {
        this.close();
        const modal = document.createElement('div');
        modal.className = 'help-modal';
        modal.id = 'explain-panel';
        modal.innerHTML = `
            <div class="help-modal-overlay"></div>
            <div class="help-modal-content" style="max-width: 720px;">
                <div class="help-modal-header">
                    <h3>${explanation.title || 'Chart Explanation'}</h3>
                    <button class="help-modal-close" id="explain-panel-close">
                        <img src="https://cdn-icons-png.flaticon.com/512/1828/1828842.png" alt="Close" style="width: 18px; height: 18px;">
                    </button>
                </div>
                <div class="help-modal-body">
                    ${this._section('Purpose', explanation.purpose)}
                    ${this._section('Data Fields', (explanation.fields || []).join(', '))}
                    ${this._section('Model', explanation.model || 'N/A')}
                    ${this._section('Equation', explanation.equation || 'Not applicable')}
                    ${this._section('Significance', explanation.significance || 'Interprets how this chart relates to risk or physical properties.')}
                    ${explanation.missing && explanation.missing.length ? this._section('Missing to unlock full view', explanation.missing.join(', ')) : ''}
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        const closeBtn = modal.querySelector('#explain-panel-close');
        const overlay = modal.querySelector('.help-modal-overlay');
        const close = () => this.close();
        closeBtn?.addEventListener('click', close);
        overlay?.addEventListener('click', close);
        this.active = modal;
        setTimeout(() => modal.classList.add('open'), 10);
    }

    close() {
        if (this.active) {
            this.active.remove();
            this.active = null;
        }
    }

    _section(label, body) {
        if (!body) return '';
        return `
            <div class="help-section">
                <h4>${label}</h4>
                <p>${body}</p>
            </div>
        `;
    }
}

window.explainPanel = new ExplainPanel();
