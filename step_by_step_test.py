#!/usr/bin/env python3
"""
Step-by-step test to identify what causes the 403 error.
"""

import time
import subprocess
import threading
from pathlib import Path
from flask import Flask, render_template, request, jsonify

def test_step_by_step():
    """Test Flask app step by step."""
    print("Step-by-step Flask test...")
    
    # Step 1: Basic Flask app (we know this works)
    print("\nStep 1: Basic Flask app")
    app1 = Flask(__name__)
    
    @app1.route('/')
    def hello1():
        return "Step 1: Basic Flask works!"
    
    success1 = test_flask_app(app1, 5002, "Step 1")
    if not success1:
        print("❌ Step 1 failed - basic Flask doesn't work")
        return False
    
    # Step 2: Add template folder configuration
    print("\nStep 2: Add template folder")
    app2 = Flask(__name__)
    
    # Set template folder like our implementation
    current_dir = Path(__file__).resolve().parent
    template_dir = current_dir / 'pyslidemorpher' / 'templates'
    if template_dir.exists():
        app2.template_folder = str(template_dir.resolve())
    
    @app2.route('/')
    def hello2():
        return "Step 2: Template folder configured!"
    
    success2 = test_flask_app(app2, 5003, "Step 2")
    if not success2:
        print("❌ Step 2 failed - template folder configuration causes issues")
        return False
    
    # Step 3: Add render_template (but return plain text on error)
    print("\nStep 3: Add template rendering")
    app3 = Flask(__name__)
    if template_dir.exists():
        app3.template_folder = str(template_dir.resolve())
    
    @app3.route('/')
    def hello3():
        try:
            return render_template('control_panel.html')
        except Exception as e:
            return f"Step 3: Template error: {str(e)}"
    
    success3 = test_flask_app(app3, 5004, "Step 3")
    if not success3:
        print("❌ Step 3 failed - template rendering causes issues")
        return False
    
    print("🎉 All steps passed!")
    return True

def test_flask_app(app, port, step_name):
    """Test a Flask app on a specific port."""
    def run_server():
        app.run(host='localhost', port=port, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(2)
    
    try:
        # Test the route
        status_result = subprocess.run(['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', f'http://localhost:{port}/'], 
                                     capture_output=True, text=True, timeout=10)
        status = int(status_result.stdout.strip())
        print(f"{step_name} status: {status}")
        
        if status == 200:
            # Get content to see what was returned
            content_result = subprocess.run(['curl', '-s', f'http://localhost:{port}/'], 
                                          capture_output=True, text=True, timeout=10)
            print(f"{step_name} content: {content_result.stdout[:100]}...")
            return True
        else:
            print(f"❌ {step_name} failed with status {status}")
            return False
            
    except Exception as e:
        print(f"❌ {step_name} failed with exception: {e}")
        return False

if __name__ == "__main__":
    test_step_by_step()