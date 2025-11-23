"""
Test All Dashboards with Real Data
Verifies that each dashboard loads data correctly without interference.
"""

import os
import sys
import requests
import json
from pathlib import Path

# Add backend to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'backend'))

BASE_URL = "http://localhost:5001"

# Dashboard configurations
DASHBOARDS = {
    'astronomy': [
        'overview',
        'star-explorer',
        'star-age',
        'sky-map',
        'light-curve',
        'clusters',
        'anomalies',
        'sky-network',
        'ml-models'
    ],
    'finance': [
        'risk',
        'streaming',
        'correlation',
        'portfolio',
        'compliance',
        'stock-explorer',
        'future-outcomes',
        'marketing-analytics',
        'ml-models',
        'game-theory'
    ]
}

# API endpoints to test for each dashboard
ENDPOINTS = {
    'astronomy': {
        'overview': ['/api/astronomy/dashboard/kpi', '/api/astronomy/dashboard/trends'],
        'star-explorer': ['/api/astronomy/color_period', '/api/astronomy/star_scatter'],
        'star-age': ['/api/astronomy/star_age'],
        'sky-map': ['/api/astronomy/sky_map'],
        'light-curve': ['/api/astronomy/star_table'],
        'clusters': ['/api/astronomy/cluster'],
        'anomalies': ['/api/astronomy/anomaly'],
        'sky-network': ['/api/astronomy/embedding'],
        'ml-models': ['/api/astronomy/color_period', '/api/astronomy/star_age']
    },
    'finance': {
        'risk': ['/api/finance/risk_kpis', '/api/finance/risk'],
        'streaming': ['/api/finance/streaming'],
        'correlation': ['/api/finance/correlation_network'],
        'portfolio': ['/api/finance/portfolio_stats'],
        'compliance': ['/api/finance/risk'],
        'stock-explorer': ['/api/finance/stock_explore'],
        'future-outcomes': ['/api/finance/future_outcomes'],
        'marketing-analytics': ['/api/finance/marketing/signage', '/api/finance/marketing/omni_channel'],
        'ml-models': ['/api/finance/risk', '/api/finance/correlation_network'],
        'game-theory': ['/api/finance/game-theory/nash-equilibrium', '/api/finance/game-theory/shapley-value']
    }
}


def test_server_running():
    """Check if Flask server is running."""
    try:
        response = requests.get(f"{BASE_URL}/api/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def test_file_listing(domain, dashboard):
    """Test file listing endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/api/{domain}/data/files?dashboard={dashboard}", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                'success': True,
                'files_count': len(data.get('files', [])),
                'active_file': data.get('active_file', {}).get('name', 'None')
            }
        return {'success': False, 'error': f'HTTP {response.status_code}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_endpoint(endpoint, dashboard=None):
    """Test a single API endpoint."""
    try:
        url = f"{BASE_URL}{endpoint}"
        if dashboard:
            url += f"?dashboard={dashboard}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Check if data is not empty
            has_data = False
            if isinstance(data, dict):
                # Check for common data keys
                for key in ['data', 'points', 'tickers', 'scenarios', 'nodes', 'values']:
                    if key in data and data[key]:
                        has_data = True
                        break
                # Check if dict has meaningful content
                if not has_data and len(data) > 0:
                    has_data = any(v for v in data.values() if v)
            elif isinstance(data, list):
                has_data = len(data) > 0
            
            return {
                'success': True,
                'has_data': has_data,
                'response_size': len(str(data))
            }
        else:
            return {
                'success': False,
                'error': f'HTTP {response.status_code}',
                'response': response.text[:200]
            }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def test_dashboard(domain, dashboard):
    """Test all endpoints for a dashboard."""
    print(f"\n  Testing {domain}/{dashboard}...")
    
    results = {
        'dashboard': dashboard,
        'domain': domain,
        'file_listing': None,
        'endpoints': {}
    }
    
    # Test file listing
    print(f"    📁 File listing...", end=' ')
    file_result = test_file_listing(domain, dashboard)
    results['file_listing'] = file_result
    if file_result['success']:
        print(f"✅ ({file_result['files_count']} files, active: {file_result['active_file']})")
    else:
        print(f"❌ {file_result.get('error', 'Unknown error')}")
    
    # Test endpoints
    endpoints = ENDPOINTS.get(domain, {}).get(dashboard, [])
    for endpoint in endpoints:
        print(f"    🔗 {endpoint}...", end=' ')
        result = test_endpoint(endpoint, dashboard)
        results['endpoints'][endpoint] = result
        if result['success']:
            data_status = "✅" if result.get('has_data') else "⚠️ (empty)"
            print(f"{data_status}")
        else:
            print(f"❌ {result.get('error', 'Unknown error')}")
    
    return results


def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 GalaxyScape X - Dashboard Testing")
    print("=" * 70)
    
    # Check if server is running
    print("\n🔍 Checking if Flask server is running...")
    if not test_server_running():
        print("❌ Flask server is not running!")
        print("   Please start the server with:")
        print("   cd backend && python3 -m flask --app api.app run --port 5001")
        return
    
    print("✅ Flask server is running")
    
    # Test all dashboards
    all_results = {}
    
    for domain, dashboards in DASHBOARDS.items():
        print(f"\n{'='*70}")
        print(f"Testing {domain.upper()} Dashboards")
        print(f"{'='*70}")
        
        domain_results = {}
        for dashboard in dashboards:
            result = test_dashboard(domain, dashboard)
            domain_results[dashboard] = result
        all_results[domain] = domain_results
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Test Summary")
    print(f"{'='*70}")
    
    total_dashboards = 0
    successful_dashboards = 0
    total_endpoints = 0
    successful_endpoints = 0
    
    for domain, dashboards in all_results.items():
        print(f"\n{domain.upper()}:")
        for dashboard, result in dashboards.items():
            total_dashboards += 1
            file_ok = result['file_listing'] and result['file_listing']['success']
            endpoints_ok = all(
                ep_result.get('success', False) 
                for ep_result in result['endpoints'].values()
            )
            
            if file_ok and endpoints_ok:
                successful_dashboards += 1
                status = "✅"
            elif file_ok or endpoints_ok:
                status = "⚠️"
            else:
                status = "❌"
            
            print(f"  {status} {dashboard}")
            
            # Count endpoints
            for endpoint, ep_result in result['endpoints'].items():
                total_endpoints += 1
                if ep_result.get('success'):
                    successful_endpoints += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {successful_dashboards}/{total_dashboards} dashboards OK")
    print(f"         {successful_endpoints}/{total_endpoints} endpoints OK")
    print(f"{'='*70}")
    
    # Save results to file
    results_file = BASE_DIR / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {results_file}")


if __name__ == '__main__':
    main()

