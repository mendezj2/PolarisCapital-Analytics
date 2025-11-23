#!/usr/bin/env python3
"""
Find Unused Files Script
Scans the codebase to identify files that aren't being imported or referenced.
"""
import os
import re
from pathlib import Path
from collections import defaultdict

def get_all_python_files(root_dir):
    """Get all Python files in the project."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip common ignore directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env']]
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def get_all_js_files(root_dir):
    """Get all JavaScript files in the project."""
    js_files = []
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '.venv', 'venv', 'env']]
        for file in files:
            if file.endswith('.js'):
                js_files.append(os.path.join(root, file))
    return js_files

def extract_imports(file_path):
    """Extract all imports from a Python file."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Match import statements
            patterns = [
                r'^import\s+([^\s]+)',
                r'^from\s+([^\s]+)\s+import',
                r'import\s+([^\s]+)',
                r'from\s+([^\s]+)\s+import'
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    imports.add(match.split('.')[0])
    except Exception as e:
        pass
    return imports

def extract_js_imports(file_path):
    """Extract imports from JavaScript files."""
    imports = set()
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            # Match require and import statements
            patterns = [
                r"require\(['\"]([^'\"]+)['\"]\)",
                r"import\s+.*from\s+['\"]([^'\"]+)['\"]",
                r"import\s+['\"]([^'\"]+)['\"]"
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content)
                imports.update(matches)
    except Exception as e:
        pass
    return imports

def normalize_module_name(file_path, root_dir):
    """Convert file path to module name."""
    rel_path = os.path.relpath(file_path, root_dir)
    # Remove .py extension
    if rel_path.endswith('.py'):
        rel_path = rel_path[:-3]
    # Convert to module format
    module_name = rel_path.replace(os.sep, '.')
    # Remove __init__ suffix
    if module_name.endswith('.__init__'):
        module_name = module_name[:-9]
    return module_name

def find_unused_files(root_dir):
    """Find files that aren't imported or referenced."""
    root_path = Path(root_dir)
    
    # Get all files
    python_files = get_all_python_files(root_dir)
    js_files = get_all_js_files(root_dir)
    
    # Build import graph
    all_imports = set()
    file_imports = {}
    
    print("Scanning Python files for imports...")
    for py_file in python_files:
        imports = extract_imports(py_file)
        file_imports[py_file] = imports
        all_imports.update(imports)
    
    print("Scanning JavaScript files for imports...")
    for js_file in js_files:
        imports = extract_js_imports(js_file)
        file_imports[js_file] = imports
    
    # Find entry points (files that are likely used)
    entry_points = set()
    entry_patterns = [
        'app.py', 'main.py', 'run.py', 'start.sh', 'index.html',
        'main.js', 'layout_manager.js'
    ]
    
    for file in python_files + js_files:
        filename = os.path.basename(file)
        if any(pattern in filename for pattern in entry_patterns):
            entry_points.add(file)
    
    # Find files that are imported
    imported_files = set()
    for file, imports in file_imports.items():
        for imp in imports:
            # Try to find matching file
            for py_file in python_files:
                module_name = normalize_module_name(py_file, root_dir)
                if imp in module_name or module_name.endswith(imp):
                    imported_files.add(py_file)
    
    # Find unused files
    unused_files = []
    for py_file in python_files:
        filename = os.path.basename(py_file)
        module_name = normalize_module_name(py_file, root_dir)
        
        # Skip if it's an entry point
        if py_file in entry_points:
            continue
        
        # Skip __init__.py files (they're often auto-loaded)
        if filename == '__init__.py':
            continue
        
        # Check if it's imported
        is_imported = False
        for other_file, imports in file_imports.items():
            if other_file == py_file:
                continue
            for imp in imports:
                if imp in module_name or module_name.endswith(imp.replace('.', os.sep)):
                    is_imported = True
                    break
            if is_imported:
                break
        
        # Check if it's referenced in HTML or config files
        is_referenced = False
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(('.html', '.json', '.md', '.sh')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            if filename in content or module_name in content:
                                is_referenced = True
                                break
                    except:
                        pass
            if is_referenced:
                break
        
        if not is_imported and not is_referenced:
            unused_files.append(py_file)
    
    return unused_files, python_files, js_files

def main():
    root_dir = Path(__file__).parent
    print("=" * 60)
    print("Finding Unused Files")
    print("=" * 60)
    
    unused_files, all_python, all_js = find_unused_files(str(root_dir))
    
    print(f"\nTotal Python files: {len(all_python)}")
    print(f"Total JavaScript files: {len(all_js)}")
    print(f"\nPotentially unused Python files: {len(unused_files)}")
    
    if unused_files:
        print("\n" + "=" * 60)
        print("UNUSED FILES:")
        print("=" * 60)
        for file in sorted(unused_files):
            rel_path = os.path.relpath(file, root_dir)
            print(f"  {rel_path}")
    else:
        print("\n✅ No unused files found (or all files are referenced)")
    
    # Also list data files
    print("\n" + "=" * 60)
    print("DATA FILES:")
    print("=" * 60)
    data_dir = root_dir / 'data'
    if data_dir.exists():
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, root_dir)
                size = os.path.getsize(file_path) / 1024  # KB
                print(f"  {rel_path} ({size:.1f} KB)")

if __name__ == '__main__':
    main()




