#!/usr/bin/env python3
"""
Performance comparison test between NumPy and PyTorch implementations.
This script measures the performance difference when using PyTorch GPU acceleration.
"""

import sys
import time
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
import subprocess

def create_test_images(num_images=4, size=(800, 600)):
    """Create test images for performance testing."""
    test_dir = Path("pytorch_test_images")
    test_dir.mkdir(exist_ok=True)
    
    colors = [
        (255, 100, 100),  # Red-ish
        (100, 255, 100),  # Green-ish
        (100, 100, 255),  # Blue-ish
        (255, 255, 100),  # Yellow-ish
    ]
    
    width, height = size
    
    for i in range(num_images):
        color = colors[i % len(colors)]
        
        # Create a complex gradient image for more realistic performance testing
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for y in range(height):
            for x in range(width):
                # Create complex patterns to stress the rendering system
                factor_x = x / width
                factor_y = y / height
                
                # Add some noise and patterns
                noise = np.random.randint(-20, 20, 3)
                pattern = np.sin(factor_x * 10) * np.cos(factor_y * 10) * 50
                
                r = int(np.clip(color[0] * factor_x + pattern + noise[0], 0, 255))
                g = int(np.clip(color[1] * factor_y + pattern + noise[1], 0, 255))
                b = int(np.clip(color[2] * (factor_x + factor_y) / 2 + pattern + noise[2], 0, 255))
                
                img[y, x] = [r, g, b]
        
        # Add some geometric shapes for visual complexity
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 6
        
        # Draw multiple circles with different colors
        for j in range(3):
            cv2.circle(img, (center_x + j*50, center_y + j*30), radius - j*20, 
                      (255-j*50, 100+j*50, 150+j*25), -1)
        
        filename = test_dir / f"test_image_{i+1:02d}.jpg"
        cv2.imwrite(str(filename), img)
        print(f"Created {filename}")
    
    return test_dir

def run_performance_test(test_dir, use_pytorch=False, test_name=""):
    """Run a performance test with the given configuration."""
    print(f"\n{'='*50}")
    print(f"Running {test_name}")
    print(f"{'='*50}")
    
    cmd = [
        sys.executable, "pyslidemorpher.py",
        str(test_dir),
        "--fps", "30",
        "--seconds-per-transition", "2.0",
        "--hold", "0.5",
        "--pixel-size", "4",
        "--size", "800x600",
        "--transition", "swarm",  # Use a computationally intensive transition
        "--log-level", "INFO"
    ]
    
    if use_pytorch:
        cmd.append("--use-pytorch")
    
    # Add output file
    output_file = f"performance_test_{'pytorch' if use_pytorch else 'numpy'}.mp4"
    cmd.extend(["--out", output_file])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        end_time = time.time()
        
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✓ Test completed successfully in {duration:.2f} seconds")
            
            # Get file size for additional metrics
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
                print(f"  Output file: {output_file} ({file_size:.1f} MB)")
                
                # Clean up output file
                os.remove(output_file)
            
            return duration, True
        else:
            print(f"✗ Test failed with return code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return duration, False
            
    except subprocess.TimeoutExpired:
        print("✗ Test timed out after 120 seconds")
        return 120.0, False
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"✗ Test failed with exception: {e}")
        return duration, False

def main():
    print("PyTorch Performance Comparison Test")
    print("=" * 50)
    
    # Check if PyTorch is available
    try:
        import torch
        pytorch_available = True
        if torch.cuda.is_available():
            print(f"✓ PyTorch with CUDA available (GPU: {torch.cuda.get_device_name()})")
        else:
            print("✓ PyTorch available (CPU only)")
    except ImportError:
        pytorch_available = False
        print("✗ PyTorch not available")
        print("To install PyTorch, run: pip install torch torchvision")
        return
    
    # Create test images
    print("\nCreating test images...")
    test_dir = create_test_images()
    
    try:
        # Run NumPy test
        numpy_time, numpy_success = run_performance_test(
            test_dir, use_pytorch=False, test_name="NumPy Implementation Test"
        )
        
        if not numpy_success:
            print("NumPy test failed, skipping PyTorch test")
            return
        
        # Run PyTorch test
        pytorch_time, pytorch_success = run_performance_test(
            test_dir, use_pytorch=True, test_name="PyTorch Implementation Test"
        )
        
        # Compare results
        print(f"\n{'='*50}")
        print("PERFORMANCE COMPARISON RESULTS")
        print(f"{'='*50}")
        
        print(f"NumPy implementation:    {numpy_time:.2f} seconds")
        
        if pytorch_success:
            print(f"PyTorch implementation:  {pytorch_time:.2f} seconds")
            
            if pytorch_time < numpy_time:
                speedup = numpy_time / pytorch_time
                print(f"\n🚀 PyTorch is {speedup:.2f}x FASTER than NumPy!")
                print(f"   Time saved: {numpy_time - pytorch_time:.2f} seconds ({((numpy_time - pytorch_time) / numpy_time * 100):.1f}%)")
            elif pytorch_time > numpy_time:
                slowdown = pytorch_time / numpy_time
                print(f"\n⚠️  PyTorch is {slowdown:.2f}x slower than NumPy")
                print(f"   This might be due to GPU memory transfer overhead for small datasets")
                print(f"   Try with larger images or more complex transitions for better GPU utilization")
            else:
                print(f"\n📊 Both implementations performed similarly")
        else:
            print("PyTorch test failed, cannot compare performance")
        
        print(f"\n{'='*50}")
        print("RECOMMENDATIONS")
        print(f"{'='*50}")
        
        if pytorch_success and pytorch_time < numpy_time:
            print("✓ Use --use-pytorch flag for better performance")
            print("✓ PyTorch acceleration is beneficial for your system")
        else:
            print("• PyTorch may not provide benefits for small images/short transitions")
            print("• Consider using PyTorch for larger images (>1920x1080) or longer transitions")
            print("• GPU acceleration is most beneficial with high pixel counts")
        
        print("\nUsage examples:")
        print("  # Use PyTorch acceleration:")
        print(f"  python pyslidemorpher.py {test_dir} --use-pytorch --realtime")
        print("  # Use NumPy (default):")
        print(f"  python pyslidemorpher.py {test_dir} --realtime")
        
    finally:
        # Clean up test images
        import shutil
        if test_dir.exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up {test_dir}")

if __name__ == "__main__":
    main()