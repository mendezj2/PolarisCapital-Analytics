#!/usr/bin/env python3
"""Run GalaxyScape X application."""
import os
import sys

# Add backend to path
project_root = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(project_root, 'backend')
sys.path.insert(0, backend_path)

# Create uploads directory
os.makedirs('uploads/astronomy', exist_ok=True)
os.makedirs('uploads/finance', exist_ok=True)

# Import and run app
from api.app import app

if __name__ == '__main__':
    port = 5001
    print(f"Starting GalaxyScape X on http://localhost:{port}")
    print("Project structure: backend/ (Python) + frontend/ (static files)")
    app.run(host='0.0.0.0', port=port, debug=True)
