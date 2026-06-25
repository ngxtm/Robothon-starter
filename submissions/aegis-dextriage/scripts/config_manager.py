import json
import sys
import os
import importlib.util

def load_config(config_path: str) -> dict:
    """Load configuration from a single JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def check_dependencies(requirements_path: str):
    """Pre-execution dependency checks."""
    if not os.path.exists(requirements_path):
        print(f"Error: Requirements file not found at {requirements_path}", file=sys.stderr)
        sys.exit(1)
        
    missing_deps = []
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Simple parsing, assuming package==version or just package
            pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
            
            # Use importlib to check if package is available
            # Note: This is a basic check. Some package names differ from import names.
            # For a more robust check, pkg_resources or importlib.metadata could be used.
            # But standard library importlib.util.find_spec is the simplest native way.
            
            # Handle common mismatches
            import_name = pkg_name
            if pkg_name.lower() == 'opencv-python':
                import_name = 'cv2'
            elif pkg_name.lower() == 'pyyaml':
                import_name = 'yaml'
                
            try:
                spec = importlib.util.find_spec(import_name)
                if spec is None:
                    missing_deps.append(pkg_name)
            except Exception:
                missing_deps.append(pkg_name)
                
    if missing_deps:
        print(f"Error: Missing required dependencies: {', '.join(missing_deps)}", file=sys.stderr)
        sys.exit(1)
    
    return True
