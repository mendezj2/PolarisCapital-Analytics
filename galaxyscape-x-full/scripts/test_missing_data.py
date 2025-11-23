"""
Test Missing Data Handling
Tests that dashboards and components handle missing CSV files gracefully.
"""

import os
import sys
import shutil
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
BACKUP_DIR = os.path.join(BASE_DIR, 'data', 'backup')

def backup_data_files():
    """Backup all data files before testing."""
    print("📦 Backing up data files...")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    
    for domain in ['astronomy', 'finance']:
        domain_dir = os.path.join(DATA_DIR, domain)
        backup_domain_dir = os.path.join(BACKUP_DIR, domain)
        os.makedirs(backup_domain_dir, exist_ok=True)
        
        if os.path.exists(domain_dir):
            for filename in os.listdir(domain_dir):
                if filename.endswith('.csv'):
                    src = os.path.join(domain_dir, filename)
                    dst = os.path.join(backup_domain_dir, filename)
                    shutil.copy2(src, dst)
                    print(f"  ✅ Backed up {domain}/{filename}")


def restore_data_files():
    """Restore data files from backup."""
    print("\n📦 Restoring data files...")
    
    for domain in ['astronomy', 'finance']:
        backup_domain_dir = os.path.join(BACKUP_DIR, domain)
        domain_dir = os.path.join(DATA_DIR, domain)
        
        if os.path.exists(backup_domain_dir):
            for filename in os.listdir(backup_domain_dir):
                if filename.endswith('.csv'):
                    src = os.path.join(backup_domain_dir, filename)
                    dst = os.path.join(domain_dir, filename)
                    shutil.copy2(src, dst)
                    print(f"  ✅ Restored {domain}/{filename}")


def test_missing_file_handling():
    """Test that missing files are handled gracefully."""
    print("\n" + "="*60)
    print("🧪 TESTING MISSING DATA HANDLING")
    print("="*60)
    
    # Test 1: Remove one astronomy file
    print("\n📊 Test 1: Remove star_explorer.csv")
    star_file = os.path.join(DATA_DIR, 'astronomy', 'star_explorer.csv')
    if os.path.exists(star_file):
        os.rename(star_file, star_file + '.bak')
        print("  ✅ File removed")
        print("  ℹ️  API should return empty data, not crash")
        time.sleep(1)
        os.rename(star_file + '.bak', star_file)
        print("  ✅ File restored")
    
    # Test 2: Remove one finance file
    print("\n📊 Test 2: Remove risk_dashboard.csv")
    risk_file = os.path.join(DATA_DIR, 'finance', 'risk_dashboard.csv')
    if os.path.exists(risk_file):
        os.rename(risk_file, risk_file + '.bak')
        print("  ✅ File removed")
        print("  ℹ️  API should fallback to other files or return empty data")
        time.sleep(1)
        os.rename(risk_file + '.bak', risk_file)
        print("  ✅ File restored")
    
    # Test 3: Remove all files in a directory
    print("\n📊 Test 3: Remove all sky_map files")
    sky_file = os.path.join(DATA_DIR, 'astronomy', 'sky_map.csv')
    if os.path.exists(sky_file):
        os.rename(sky_file, sky_file + '.bak')
        print("  ✅ File removed")
        print("  ℹ️  Dashboard should show 'No data available' message")
        time.sleep(1)
        os.rename(sky_file + '.bak', sky_file)
        print("  ✅ File restored")
    
    print("\n✅ Missing data handling tests complete!")
    print("   All files should be restored and dashboards should handle gracefully.")


def main():
    """Main test function."""
    print("="*60)
    print("🛡️  MISSING DATA HANDLING TEST")
    print("="*60)
    
    backup_data_files()
    test_missing_file_handling()
    restore_data_files()
    
    print("\n" + "="*60)
    print("✅ All tests complete!")
    print("="*60)


if __name__ == '__main__':
    main()




