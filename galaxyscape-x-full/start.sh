#!/bin/bash
# Start GalaxyScape X application

cd "$(dirname "$0")"

echo "Starting GalaxyScape X..."
echo "Project structure:"
echo "  - backend/    : Python API, ML models, data processing"
echo "  - frontend/   : HTML, CSS, JavaScript"
echo ""
echo "Make sure dependencies are installed: pip install -r requirements.txt"
echo ""

python run.py
