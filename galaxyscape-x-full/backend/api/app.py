"""Main Flask application for GalaxyScape X."""
from flask import Flask
import sys
import os

# Add backend to path
backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_path)

from api.astronomy_api import astronomy_bp
from api.finance_api import finance_bp

def create_app():
    # Point to frontend/static directory
    frontend_static = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'frontend', 'static')
    app = Flask(__name__, static_folder=frontend_static, static_url_path='')
    app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
    
    app.register_blueprint(astronomy_bp, url_prefix='/api/astronomy')
    app.register_blueprint(finance_bp, url_prefix='/api/finance')
    
    @app.route('/')
    def index():
        return app.send_static_file('index.html')
    
    @app.route('/health')
    @app.route('/api/health')
    def api_health():
        return {'status': 'ok', 'message': 'Galaxyscape X online'}
    
    return app

app = create_app()

if __name__ == '__main__':
    uploads_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(os.path.join(uploads_dir, 'astronomy'), exist_ok=True)
    os.makedirs(os.path.join(uploads_dir, 'finance'), exist_ok=True)
    app.run(host='0.0.0.0', port=5001, debug=True)
