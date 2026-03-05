#!/usr/bin/env python3
"""
Test to see if importing web_gui module causes the 403 issue.
"""

import time
import subprocess
import threading

def test_direct_import():
    """Test importing web_gui directly."""
    print("Testing direct import of web_gui...")
    
    try:
        # Import our web_gui module
        from pyslidemorpher.web_gui import start_web_server, FLASK_AVAILABLE
        
        if not FLASK_AVAILABLE:
            print("Flask not available")
            return False
        
        print("✓ web_gui module imported successfully")
        
        # Start the web server using our module
        print("Starting web server from imported module...")
        server_thread = start_web_server(port=5005)
        
        if not server_thread:
            print("Failed to start web server")
            return False
        
        # Wait for server to start
        time.sleep(3)
        
        # Test the server
        print("Testing server...")
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5005/test'], 
                                     capture_output=True, text=True, timeout=10)
        status = int(status_result.stdout.strip())
        print(f"Server status: {status}")
        
        if status == 200:
            print("✓ Server works after direct import")
            return True
        else:
            print(f"✗ Server failed with status {status} after direct import")
            
            # Get content to see what's happening
            content_result = subprocess.run(['curl', '-s', 'http://localhost:5005/test'], 
                                          capture_output=True, text=True, timeout=10)
            print(f"Response content: {content_result.stdout[:200]}...")
            return False
            
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_without_other_imports():
    """Test Flask without importing other pyslidemorpher modules."""
    print("\nTesting Flask without other pyslidemorpher imports...")
    
    try:
        # Import Flask directly without going through pyslidemorpher
        from flask import Flask
        from pathlib import Path
        
        app = Flask(__name__)
        
        # Set template folder
        current_dir = Path(__file__).resolve().parent
        template_dir = current_dir / 'pyslidemorpher' / 'templates'
        if template_dir.exists():
            app.template_folder = str(template_dir.resolve())
        
        @app.route('/test')
        def test():
            return "Direct Flask without pyslidemorpher imports!"
        
        def run_server():
            app.run(host='localhost', port=5006, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        time.sleep(2)
        
        # Test the server
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5006/test'], 
                                     capture_output=True, text=True, timeout=10)
        status = int(status_result.stdout.strip())
        print(f"Direct Flask status: {status}")
        
        if status == 200:
            print("✓ Direct Flask works without pyslidemorpher imports")
            return True
        else:
            print(f"✗ Direct Flask failed with status {status}")
            return False
            
    except Exception as e:
        print(f"✗ Direct Flask test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing import-related issues...")
    
    # Test without other imports first
    success1 = test_without_other_imports()
    
    # Test with our module import
    success2 = test_direct_import()
    
    if success1 and not success2:
        print("🔍 Issue is caused by importing pyslidemorpher modules")
    elif not success1 and not success2:
        print("🔍 Issue is more fundamental")
    elif success1 and success2:
        print("🎉 Both tests passed - issue might be elsewhere")
    else:
        print("🤔 Unexpected results")