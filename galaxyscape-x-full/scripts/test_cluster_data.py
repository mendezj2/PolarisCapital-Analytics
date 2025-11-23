"""
Test Cluster Data File Rendering
Verifies that cluster_analysis.csv loads and renders correctly in the clusters dashboard.
"""

import os
import sys
import requests
import json
import pandas as pd
from pathlib import Path

# Add backend to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR / 'backend'))

BASE_URL = "http://localhost:5001"

def test_cluster_file_exists():
    """Test that cluster_analysis.csv exists and has data."""
    print("🔍 Testing cluster_analysis.csv file...")
    
    cluster_file = BASE_DIR / 'data' / 'raw' / 'astronomy' / 'cluster_analysis.csv'
    
    if not cluster_file.exists():
        print(f"❌ File not found: {cluster_file}")
        return False
    
    try:
        df = pd.read_csv(cluster_file)
        print(f"✅ File exists: {len(df)} rows, {len(df.columns)} columns")
        
        # Check for cluster column
        if 'cluster' in df.columns:
            unique_clusters = df['cluster'].unique()
            print(f"✅ Cluster column found: {len(unique_clusters)} unique clusters")
            print(f"   Cluster IDs: {sorted(unique_clusters)}")
            
            # Show cluster distribution
            cluster_counts = df['cluster'].value_counts().sort_index()
            print(f"   Cluster distribution:")
            for cluster_id, count in cluster_counts.items():
                print(f"     Cluster {cluster_id}: {count} stars")
        else:
            print("⚠️  No 'cluster' column found in file")
        
        # Check for required columns
        required_cols = ['temperature', 'mass', 'radius', 'luminosity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  Missing columns: {missing_cols}")
        else:
            print(f"✅ All required columns present")
        
        return True
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


def test_cluster_api():
    """Test cluster API endpoint."""
    print("\n🔍 Testing cluster API endpoint...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/astronomy/cluster?n_clusters=5", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ API responded successfully")
            print(f"   Points: {len(data.get('points', []))}")
            print(f"   Cluster labels: {len(data.get('cluster_labels', []))}")
            print(f"   Cluster centers: {len(data.get('cluster_centers', []))}")
            print(f"   Number of clusters: {data.get('n_clusters', 0)}")
            
            # Check if data is valid
            if len(data.get('points', [])) > 0:
                print(f"✅ Data is valid and non-empty")
                
                # Check cluster distribution
                if data.get('cluster_labels'):
                    from collections import Counter
                    cluster_dist = Counter(data['cluster_labels'])
                    print(f"   Cluster distribution:")
                    for cluster_id, count in sorted(cluster_dist.items()):
                        print(f"     Cluster {cluster_id}: {count} points")
                
                return True
            else:
                print("⚠️  API returned empty data")
                return False
        else:
            print(f"❌ API error: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Flask server")
        print("   Please start the server: cd backend && python3 -m flask --app api.app run --port 5001")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False


def test_file_listing():
    """Test file listing for clusters dashboard."""
    print("\n🔍 Testing file listing for clusters dashboard...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/astronomy/data/files?dashboard=clusters", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ File listing API responded")
            
            if data.get('active_file'):
                active = data['active_file']
                print(f"   Active file: {active.get('name', 'Unknown')}")
                print(f"   Rows: {active.get('row_count', 0):,}")
                print(f"   Columns: {len(active.get('columns', []))}")
            
            if data.get('available_files'):
                files = data['available_files']
                print(f"   Available files: {len(files)}")
                for file in files:
                    print(f"     - {file.get('name', 'Unknown')} ({file.get('row_count', 0):,} rows)")
                    if 'cluster_analysis.csv' in file.get('name', ''):
                        print(f"       ✅ cluster_analysis.csv found!")
            
            return True
        else:
            print(f"❌ File listing error: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing file listing: {e}")
        return False


def test_cluster_dashboard_components():
    """Test that cluster dashboard components can render."""
    print("\n🔍 Testing cluster dashboard components...")
    
    endpoints = [
        ('/api/astronomy/cluster', 'Cluster visualization'),
        ('/api/astronomy/star_table?limit=10', 'Star table'),
        ('/api/astronomy/star_scatter', 'Scatter plot')
    ]
    
    all_passed = True
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                has_data = len(data) > 0 if isinstance(data, (list, dict)) else bool(data)
                status = "✅" if has_data else "⚠️"
                print(f"   {status} {name}: {'Has data' if has_data else 'Empty response'}")
            else:
                print(f"   ❌ {name}: HTTP {response.status_code}")
                all_passed = False
        except Exception as e:
            print(f"   ❌ {name}: {e}")
            all_passed = False
    
    return all_passed


def main():
    """Main test function."""
    print("=" * 70)
    print("🧪 Cluster Data File Testing")
    print("=" * 70)
    print()
    
    results = {
        'file_exists': False,
        'api_works': False,
        'file_listing': False,
        'components': False
    }
    
    # Test 1: File exists
    results['file_exists'] = test_cluster_file_exists()
    
    # Test 2: API endpoint
    results['api_works'] = test_cluster_api()
    
    # Test 3: File listing
    results['file_listing'] = test_file_listing()
    
    # Test 4: Dashboard components
    results['components'] = test_cluster_dashboard_components()
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Test Summary")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {test_name.replace('_', ' ').title()}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ All tests passed! Cluster data file is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 70)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

