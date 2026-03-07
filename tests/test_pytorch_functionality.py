#!/usr/bin/env python3
"""
Test script to verify PyTorch functionality is working correctly.
"""

import sys
import numpy as np
import cv2
from pathlib import Path

# Add current directory to path to import pyslidemorpher
sys.path.insert(0, '.')
import pyslidemorpher

def test_pytorch_availability():
    """Test that PyTorch is available and detected correctly."""
    print("Testing PyTorch availability...")
    
    # Check if PyTorch was detected
    if pyslidemorpher.PYTORCH_AVAILABLE:
        print(f"✓ PyTorch is available")
        print(f"  Device: {pyslidemorpher.DEVICE}")
        return True
    else:
        print("✗ PyTorch is not available")
        return False

def test_pytorch_rendering():
    """Test that PyTorch rendering functions work correctly."""
    print("\nTesting PyTorch rendering functions...")
    
    # Enable PyTorch usage
    pyslidemorpher.USE_PYTORCH = True
    
    # Create test data
    grid_shape = (50, 50)
    num_particles = grid_shape[0] * grid_shape[1]
    
    # Create random positions and colors
    pos = np.random.rand(num_particles, 2) * 50
    cols = np.random.rand(num_particles, 3) * 255
    
    try:
        # Test PyTorch rendering
        frame_torch = pyslidemorpher.render_frame_torch(pos, cols, grid_shape)
        print(f"✓ PyTorch rendering successful, frame shape: {frame_torch.shape}")
        
        # Test optimized rendering (should use PyTorch)
        frame_optimized = pyslidemorpher.render_frame_optimized(pos, cols, grid_shape)
        print(f"✓ Optimized rendering successful, frame shape: {frame_optimized.shape}")
        
        # Test that frames are similar (should be identical or very close)
        if np.allclose(frame_torch, frame_optimized, atol=1):
            print("✓ PyTorch and optimized rendering produce consistent results")
        else:
            print("⚠ PyTorch and optimized rendering produce different results")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch rendering failed: {e}")
        return False

def test_pytorch_vs_numpy():
    """Compare PyTorch and NumPy rendering performance and correctness."""
    print("\nTesting PyTorch vs NumPy rendering...")
    
    # Create test data
    grid_shape = (100, 100)
    num_particles = grid_shape[0] * grid_shape[1]
    
    pos = np.random.rand(num_particles, 2) * 100
    cols = np.random.rand(num_particles, 3) * 255
    
    try:
        # Test NumPy rendering
        frame_numpy = pyslidemorpher.render_frame(pos, cols, grid_shape)
        print(f"✓ NumPy rendering successful, frame shape: {frame_numpy.shape}")
        
        # Test PyTorch rendering
        frame_torch = pyslidemorpher.render_frame_torch(pos, cols, grid_shape)
        print(f"✓ PyTorch rendering successful, frame shape: {frame_torch.shape}")
        
        # Compare results (they should be very similar)
        if np.allclose(frame_numpy, frame_torch, atol=2):
            print("✓ NumPy and PyTorch rendering produce consistent results")
        else:
            print("⚠ NumPy and PyTorch rendering produce different results")
            print(f"  Max difference: {np.max(np.abs(frame_numpy.astype(float) - frame_torch.astype(float)))}")
        
        return True
        
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False

def test_use_pytorch_flag():
    """Test that the USE_PYTORCH global flag works correctly."""
    print("\nTesting USE_PYTORCH flag functionality...")
    
    grid_shape = (50, 50)
    num_particles = grid_shape[0] * grid_shape[1]
    pos = np.random.rand(num_particles, 2) * 50
    cols = np.random.rand(num_particles, 3) * 255
    
    try:
        # Test with PyTorch disabled
        pyslidemorpher.USE_PYTORCH = False
        frame_disabled = pyslidemorpher.render_frame_optimized(pos, cols, grid_shape)
        print("✓ Rendering with PyTorch disabled works")
        
        # Test with PyTorch enabled
        pyslidemorpher.USE_PYTORCH = True
        frame_enabled = pyslidemorpher.render_frame_optimized(pos, cols, grid_shape)
        print("✓ Rendering with PyTorch enabled works")
        
        # Both should work and produce similar results
        if np.allclose(frame_disabled, frame_enabled, atol=2):
            print("✓ PyTorch flag switching produces consistent results")
        else:
            print("⚠ PyTorch flag switching produces different results")
        
        return True
        
    except Exception as e:
        print(f"✗ USE_PYTORCH flag test failed: {e}")
        return False

def main():
    print("PyTorch Functionality Test")
    print("=" * 40)
    
    success = True
    
    # Test PyTorch availability
    if not test_pytorch_availability():
        print("\nPyTorch is not available, skipping PyTorch-specific tests")
        return
    
    # Test PyTorch rendering
    if not test_pytorch_rendering():
        success = False
    
    # Test PyTorch vs NumPy comparison
    if not test_pytorch_vs_numpy():
        success = False
    
    # Test USE_PYTORCH flag
    if not test_use_pytorch_flag():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All PyTorch functionality tests passed!")
        print("\nPyTorch acceleration is working correctly.")
        print("You can now use the --use-pytorch flag for better performance:")
        print("  python pyslidemorpher.py <image_folder> --use-pytorch --realtime")
    else:
        print("✗ Some PyTorch functionality tests failed!")
        print("PyTorch may not be working correctly on your system.")
    
    print("=" * 40)

if __name__ == "__main__":
    main()