#!/usr/bin/env python3
"""
Minimal Flask app test to isolate the 403 issue.
"""

import time
import subprocess
import threading
from flask import Flask

def test_minimal_flask():
    """Test a minimal Flask app."""
    print("Testing minimal Flask app...")
    
    # Create minimal Flask app
    app = Flask(__name__)
    
    @app.route('/')
    def hello():
        return "Hello, World!"
    
    @app.route('/test')
    def test():
        return "Test route works!"
    
    # Start server in thread
    def run_server():
        app.run(host='localhost', port=5001, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    print("Waiting for minimal server to start...")
    time.sleep(3)
    
    try:
        # Test the routes
        print("Testing minimal Flask routes...")
        
        # Test main route
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5001/'], 
                                     capture_output=True, text=True, timeout=10)
        main_status = int(status_result.stdout.strip())
        print(f"Minimal app main route status: {main_status}")
        
        # Test test route
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', 'http://localhost:5001/test'], 
                                     capture_output=True, text=True, timeout=10)
        test_status = int(status_result.stdout.strip())
        print(f"Minimal app test route status: {test_status}")
        
        if main_status == 200 and test_status == 200:
            print("✓ Minimal Flask app works correctly")
            return True
        else:
            print("✗ Minimal Flask app also has issues")
            return False
            
    except Exception as e:
        print(f"✗ Minimal Flask test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_minimal_flask()
    if success:
        print("🎉 Minimal Flask test passed - issue is with our implementation")
    else:
        print("❌ Minimal Flask test failed - issue is with Flask setup")