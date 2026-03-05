#!/usr/bin/env python3
"""
Debug script to test web GUI template path resolution.
"""

import sys
from pathlib import Path

def debug_template_paths():
    """Debug template path resolution."""
    print("Debugging template path resolution...")
    
    # Import the web GUI module
    from pyslidemorpher.web_gui import create_web_app, FLASK_AVAILABLE
    
    if not FLASK_AVAILABLE:
        print("Flask not available")
        return False
    
    # Get current working directory
    cwd = Path.cwd()
    print(f"Current working directory: {cwd}")
    
    # Get the web_gui.py file location
    web_gui_file = Path(__file__).parent / "pyslidemorpher" / "web_gui.py"
    print(f"web_gui.py location: {web_gui_file}")
    print(f"web_gui.py exists: {web_gui_file.exists()}")
    
    # Check template directory from web_gui.py perspective
    current_dir = web_gui_file.parent
    template_dir = current_dir / 'templates'
    print(f"Template directory (from web_gui.py): {template_dir}")
    print(f"Template directory exists: {template_dir.exists()}")
    
    # Check template file
    template_file = template_dir / 'control_panel.html'
    print(f"Template file: {template_file}")
    print(f"Template file exists: {template_file.exists()}")
    
    # Create Flask app and check its template folder
    app = create_web_app()
    if app:
        print(f"Flask app template folder: {app.template_folder}")
        print(f"Flask app template folder exists: {Path(app.template_folder).exists()}")
        
        # Test with Flask test client
        with app.test_client() as client:
            response = client.get('/')
            print(f"Test client response status: {response.status_code}")
            
        return True
    else:
        print("Failed to create Flask app")
        return False

if __name__ == "__main__":
    debug_template_paths()