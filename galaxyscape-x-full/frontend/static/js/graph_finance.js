let svg, simulation;

function renderFinanceGraph(data) {
    const container = d3.select('#finance-graph');
    container.selectAll('*').remove();
    
    svg = container.append('svg')
        .attr('width', '100%')
        .attr('height', '100%');
    
    const width = container.node().getBoundingClientRect().width;
    const height = container.node().getBoundingClientRect().height;
    
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    
    if (nodes.length === 0) return;
    
    // Create force simulation
    simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(d => d.id).distance(100))
        .force('charge', d3.forceManyBody().strength(-300))
        .force('center', d3.forceCenter(width / 2, height / 2));
    
    // Draw edges
    const link = svg.append('g')
        .selectAll('line')
        .data(edges)
        .enter()
        .append('line')
        .attr('stroke', d => d.correlation > 0 ? '#00a8e8' : '#ff6b6b')
        .attr('stroke-width', d => Math.abs(d.weight) * 5)
        .attr('opacity', 0.6);
    
    // Draw nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 8)
        .attr('fill', d => {
            const risk = d.risk_score || 50;
            if (risk < 40) return '#51cf66';
            if (risk < 70) return '#ffd43b';
            return '#ff6b6b';
        })
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 12);
            document.getElementById('risk-info').innerHTML = 
                `<div><strong>${d.id}</strong></div>
                 <div>Risk: ${d.risk_score || 50}</div>
                 <div>Sector: ${d.sector || 'N/A'}</div>`;
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 8);
        });
    
    // Add labels
    const label = svg.append('g')
        .selectAll('text')
        .data(nodes)
        .enter()
        .append('text')
        .text(d => d.id)
        .attr('font-size', '10px')
        .attr('fill', '#0c1a2a')
        .attr('dx', 10)
        .attr('dy', 4);
    
    // Update positions on tick
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);
        
        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
        
        label
            .attr('x', d => d.x)
            .attr('y', d => d.y);
    });
    
    window.renderFinanceGraph = renderFinanceGraph;
}

