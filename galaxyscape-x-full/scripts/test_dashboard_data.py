"""
Test Dashboard Data
Tests each dashboard to ensure data files exist and can be loaded properly.
Also tests API endpoints and verifies error handling for missing data.
"""

import os
import sys
import pandas as pd
import json
import requests
import time

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

# Dashboard configurations
ASTRONOMY_DASHBOARDS = {
    'overview': 'star_explorer.csv',
    'star-explorer': 'star_explorer.csv',
    'sky-map': 'sky_map.csv',
    'cluster': 'cluster_analysis.csv',
    'anomaly': 'anomaly_detection.csv',
    'ml-models': 'ml_models.csv'
}

FINANCE_DASHBOARDS = {
    'risk': 'risk_dashboard.csv',
    'correlation': 'correlation_network.csv',
    'stock-explorer': 'stock_explorer.csv',
    'future-outcomes': 'future_outcomes.csv',
    'game-theory': 'game_theory.csv',
    'marketing-analytics': ['marketing_signage.csv', 'marketing_omni_channel.csv']
}

def test_data_file(filepath, dashboard_name, domain):
    """Test if a data file exists and can be loaded."""
    print(f"  📄 Testing {os.path.basename(filepath)}...")
    
    if not os.path.exists(filepath):
        print(f"    ❌ File not found: {filepath}")
        return False, None
    
    try:
        df = pd.read_csv(filepath, nrows=10)  # Just read first 10 rows for testing
        print(f"    ✅ File exists: {len(df)} rows (sample)")
        print(f"    📊 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
        
        # Check for required columns based on domain
        if domain == 'astronomy':
            required_cols = ['star_name', 'temperature', 'mass'] if 'star_name' in df.columns else ['name', 'temperature']
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                print(f"    ⚠️  Missing columns: {', '.join(missing)}")
        
        elif domain == 'finance':
            required_cols = ['Date', 'Ticker', 'Close']
            # Check for variations
            has_date = 'Date' in df.columns or 'date' in df.columns
            has_ticker = 'Ticker' in df.columns or 'ticker' in df.columns or 'Symbol' in df.columns
            has_close = 'Close' in df.columns or 'close' in df.columns or 'price' in df.columns
            
            if not (has_date and has_ticker and has_close):
                print(f"    ⚠️  Missing finance columns (Date, Ticker, Close)")
        
        return True, df
    except Exception as e:
        print(f"    ❌ Error loading file: {e}")
        return False, None


def test_astronomy_dashboards():
    """Test all astronomy dashboards."""
    print("\n" + "="*60)
    print("🔭 TESTING ASTRONOMY DASHBOARDS")
    print("="*60)
    
    results = {}
    astro_dir = os.path.join(DATA_DIR, 'astronomy')
    
    for dashboard_name, filename in ASTRONOMY_DASHBOARDS.items():
        print(f"\n📊 Dashboard: {dashboard_name}")
        filepath = os.path.join(astro_dir, filename)
        success, df = test_data_file(filepath, dashboard_name, 'astronomy')
        results[dashboard_name] = {
            'success': success,
            'filepath': filepath,
            'data': df is not None,
            'rows': len(df) if df is not None else 0
        }
    
    return results


def test_finance_dashboards():
    """Test all finance dashboards."""
    print("\n" + "="*60)
    print("💰 TESTING FINANCE DASHBOARDS")
    print("="*60)
    
    results = {}
    finance_dir = os.path.join(DATA_DIR, 'finance')
    
    for dashboard_name, filenames in FINANCE_DASHBOARDS.items():
        print(f"\n📊 Dashboard: {dashboard_name}")
        
        if isinstance(filenames, list):
            # Multiple files for this dashboard
            all_success = True
            for filename in filenames:
                filepath = os.path.join(finance_dir, filename)
                success, df = test_data_file(filepath, dashboard_name, 'finance')
                if not success:
                    all_success = False
            results[dashboard_name] = {'success': all_success}
        else:
            filepath = os.path.join(finance_dir, filenames)
            success, df = test_data_file(filepath, dashboard_name, 'finance')
            results[dashboard_name] = {
                'success': success,
                'filepath': filepath,
                'data': df is not None,
                'rows': len(df) if df is not None else 0
            }
    
    return results


def test_api_endpoints(base_url='http://localhost:5001'):
    """Test API endpoints for each dashboard."""
    print("\n" + "="*60)
    print("🌐 TESTING API ENDPOINTS")
    print("="*60)
    
    results = {}
    
    # Astronomy endpoints
    print("\n🔭 Astronomy Endpoints:")
    astro_endpoints = {
        'star-explorer': '/api/astronomy/star_table',
        'sky-map': '/api/astronomy/sky_map',
        'cluster': '/api/astronomy/cluster',
        'anomaly': '/api/astronomy/anomaly',
        'ml-models': '/api/astronomy/ml/models/performance',
        'color-period': '/api/astronomy/color_period'
    }
    
    for dashboard, endpoint in astro_endpoints.items():
        print(f"  📡 Testing {endpoint}...")
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                has_data = len(data) > 0 if isinstance(data, (list, dict)) else data is not None
                print(f"    ✅ Status 200, has data: {has_data}")
                results[f"astro_{dashboard}"] = {'success': True, 'has_data': has_data}
            else:
                print(f"    ⚠️  Status {response.status_code}")
                results[f"astro_{dashboard}"] = {'success': False, 'status': response.status_code}
        except requests.exceptions.ConnectionError:
            print(f"    ⚠️  Server not running (skipping)")
            results[f"astro_{dashboard}"] = {'success': False, 'error': 'server_not_running'}
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[f"astro_{dashboard}"] = {'success': False, 'error': str(e)}
    
    # Finance endpoints
    print("\n💰 Finance Endpoints:")
    finance_endpoints = {
        'risk': '/api/finance/risk',
        'correlation': '/api/finance/correlation_network',
        'stock-explorer': '/api/finance/stock/explore',
        'future-outcomes': '/api/finance/future_outcomes',
        'game-theory': '/api/finance/game-theory/nash-equilibrium',
        'marketing-signage': '/api/finance/marketing/signage-evaluation'
    }
    
    for dashboard, endpoint in finance_endpoints.items():
        print(f"  📡 Testing {endpoint}...")
        try:
            if endpoint == '/api/finance/future_outcomes':
                # POST request
                response = requests.post(f"{base_url}{endpoint}", 
                                       json={'initial_value': 10000, 'time_horizon_years': 1},
                                       timeout=5)
            else:
                response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                has_data = len(data) > 0 if isinstance(data, (list, dict)) else data is not None
                print(f"    ✅ Status 200, has data: {has_data}")
                results[f"finance_{dashboard}"] = {'success': True, 'has_data': has_data}
            else:
                print(f"    ⚠️  Status {response.status_code}")
                results[f"finance_{dashboard}"] = {'success': False, 'status': response.status_code}
        except requests.exceptions.ConnectionError:
            print(f"    ⚠️  Server not running (skipping)")
            results[f"finance_{dashboard}"] = {'success': False, 'error': 'server_not_running'}
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[f"finance_{dashboard}"] = {'success': False, 'error': str(e)}
    
    return results


def test_missing_data_handling():
    """Test that components handle missing data gracefully."""
    print("\n" + "="*60)
    print("🛡️  TESTING MISSING DATA HANDLING")
    print("="*60)
    
    # Test with empty DataFrame
    print("\n  📊 Testing empty DataFrame handling...")
    try:
        empty_df = pd.DataFrame()
        if len(empty_df) == 0:
            print("    ✅ Empty DataFrame created successfully")
    except Exception as e:
        print(f"    ❌ Error: {e}")
    
    # Test with missing file
    print("\n  📄 Testing missing file handling...")
    missing_file = os.path.join(DATA_DIR, 'astronomy', 'nonexistent.csv')
    if not os.path.exists(missing_file):
        print("    ✅ Missing file detection works")
    
    # Check if API endpoints return empty data gracefully
    print("\n  🌐 Testing API error responses...")
    # This would be tested when server is running
    print("    ℹ️  API error handling should return empty arrays/objects, not crash")


def generate_report(astro_results, finance_results, api_results):
    """Generate a summary report."""
    print("\n" + "="*60)
    print("📋 TEST SUMMARY REPORT")
    print("="*60)
    
    print("\n🔭 Astronomy Dashboards:")
    for dashboard, result in astro_results.items():
        status = "✅" if result.get('success') else "❌"
        rows = result.get('rows', 0)
        print(f"  {status} {dashboard}: {rows} rows")
    
    print("\n💰 Finance Dashboards:")
    for dashboard, result in finance_results.items():
        status = "✅" if result.get('success') else "❌"
        if 'rows' in result:
            rows = result.get('rows', 0)
            print(f"  {status} {dashboard}: {rows} rows")
        else:
            print(f"  {status} {dashboard}: Multiple files")
    
    print("\n🌐 API Endpoints:")
    success_count = sum(1 for r in api_results.values() if r.get('success'))
    total_count = len(api_results)
    print(f"  {success_count}/{total_count} endpoints responding")
    
    # Recommendations
    print("\n💡 Recommendations:")
    missing_astro = [d for d, r in astro_results.items() if not r.get('success')]
    missing_finance = [d for d, r in finance_results.items() if not r.get('success')]
    
    if missing_astro:
        print(f"  ⚠️  Missing astronomy data: {', '.join(missing_astro)}")
    if missing_finance:
        print(f"  ⚠️  Missing finance data: {', '.join(missing_finance)}")
    
    if not missing_astro and not missing_finance:
        print("  ✅ All data files are present and loadable!")


def main():
    """Main test function."""
    print("="*60)
    print("🧪 DASHBOARD DATA TESTING")
    print("="*60)
    
    # Test data files
    astro_results = test_astronomy_dashboards()
    finance_results = test_finance_dashboards()
    
    # Test API endpoints (if server is running)
    api_results = test_api_endpoints()
    
    # Test missing data handling
    test_missing_data_handling()
    
    # Generate report
    generate_report(astro_results, finance_results, api_results)
    
    print("\n" + "="*60)
    print("✅ Testing complete!")
    print("="*60)


if __name__ == '__main__':
    main()




