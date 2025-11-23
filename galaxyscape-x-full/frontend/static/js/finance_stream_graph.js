/* Finance streaming graph updates. */
let graphData = { nodes: [], edges: [] };
let updateInterval = null;

export function initializeStreamGraph(options = {}) {
    const updateMethod = options.updateMethod || 'polling';
    const pollIntervalMs = options.pollIntervalMs || 2000;
    
    if (updateMethod === 'websocket') {
        connectWebSocket();
    } else {
        startPolling(pollIntervalMs);
    }
}

function connectWebSocket() {
    console.log('WebSocket connection not implemented - using polling');
    startPolling(2000);
}

function startPolling(intervalMs) {
    updateInterval = setInterval(fetchLatestGraphData, intervalMs);
    fetchLatestGraphData();
}

async function fetchLatestGraphData() {
    try {
        const response = await fetch('/api/finance/stream/graph');
        const data = await response.json();
        updateGraphIncremental(data);
    } catch (error) {
        console.error('Failed to fetch graph data', error);
    }
}

function updateGraphIncremental(newData) {
    graphData = newData;
    
    // Trigger graph re-render if function exists
    if (window.renderFinanceGraph) {
        window.renderFinanceGraph(newData);
    }
}

export function updateNodeState(nodeId, riskScore, volatility, anomalyScore) {
    const node = graphData.nodes.find(n => n.id === nodeId);
    if (node) {
        node.risk_score = riskScore;
        node.volatility = volatility;
        node.anomaly_score = anomalyScore;
    }
}

export function stopStreaming() {
    if (updateInterval) {
        clearInterval(updateInterval);
        updateInterval = null;
    }
}

