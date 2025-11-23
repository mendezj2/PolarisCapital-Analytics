let svg, simulation;

function renderAstronomyGraph(data) {
    const container = d3.select('#astronomy-graph');
    container.selectAll('*').remove();
    
    svg = container.append('svg')
        .attr('width', '100%')
        .attr('height', '100%');
    
    const width = container.node().getBoundingClientRect().width;
    const height = container.node().getBoundingClientRect().height;
    
    // Scale data to fit
    const nodes = data.nodes || [];
    const edges = data.edges || [];
    
    if (nodes.length === 0) return;
    
    const xExtent = d3.extent(nodes, d => d.x);
    const yExtent = d3.extent(nodes, d => d.y);
    
    const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([50, width - 50]);
    
    const yScale = d3.scaleLinear()
        .domain(yExtent)
        .range([50, height - 50]);
    
    // Draw edges
    const link = svg.append('g')
        .selectAll('line')
        .data(edges)
        .enter()
        .append('line')
        .attr('stroke', 'rgba(138, 180, 255, 0.3)')
        .attr('stroke-width', 1);
    
    // Draw nodes
    const node = svg.append('g')
        .selectAll('circle')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('r', 4)
        .attr('fill', '#8ab4ff')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 6);
            document.getElementById('star-info').textContent = 
                `Star ID: ${d.id}, Cluster: ${d.cluster}`;
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 4);
        });
    
    window.renderAstronomyGraph = renderAstronomyGraph;
}

