#!/usr/bin/env python3
"""
Test script to verify all requirements are properly installed.
Run this before training to ensure your environment is set up correctly.
"""

import sys
import importlib
from packaging import version


def test_import_and_version(package_name, min_version=None, import_name=None):
    """
    Test if a package can be imported and meets minimum version requirements.
    
    Args:
        package_name (str): Name of the package to test
        min_version (str): Minimum required version
        import_name (str): Alternative import name if different from package name
    
    Returns:
        bool: True if test passes, False otherwise
    """
    if import_name is None:
        import_name = package_name
    
    try:
        # Try to import the package
        module = importlib.import_module(import_name)
        print(f"✅ {package_name}: Successfully imported")
        
        # Check version if specified
        if min_version and hasattr(module, '__version__'):
            current_version = module.__version__
            if version.parse(current_version) >= version.parse(min_version):
                print(f"   Version: {current_version} (>= {min_version}) ✅")
                return True
            else:
                print(f"   Version: {current_version} (< {min_version}) ❌")
                return False
        elif min_version:
            print(f"   Warning: Could not verify version (no __version__ attribute)")
            return True
        else:
            print(f"   Version check skipped")
            return True
            
    except ImportError as e:
        print(f"❌ {package_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"❌ {package_name}: Unexpected error - {e}")
        return False


def test_torch_functionality():
    """Test basic PyTorch functionality"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        print("\n🔧 Testing PyTorch functionality...")
        
        # Test tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = x + y
        print("   ✅ Basic tensor operations work")
        
        # Test simple neural network
        model = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Test forward pass
        output = model(x)
        print("   ✅ Neural network forward pass works")
        
        # Test optimizer
        optimizer = optim.Adam(model.parameters())
        loss = nn.MSELoss()(output, torch.randn(2, 1))
        loss.backward()
        optimizer.step()
        print("   ✅ Backpropagation and optimization work")
        
        # Test CUDA availability (informational)
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ️  CUDA not available (CPU only)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ PyTorch functionality test failed: {e}")
        return False


def test_torchvision_functionality():
    """Test basic torchvision functionality"""
    try:
        import torchvision
        import torchvision.transforms as transforms
        import torchvision.datasets as datasets
        from torch.utils.data import DataLoader
        
        print("\n🖼️  Testing torchvision functionality...")
        
        # Test transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        print("   ✅ Transforms creation works")
        
        # Test dataset access (without downloading)
        try:
            # Just check if MNIST class exists and can be instantiated
            # (without actually downloading data)
            mnist_class = datasets.MNIST
            print("   ✅ MNIST dataset class accessible")
        except Exception as e:
            print(f"   ❌ MNIST dataset class error: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ torchvision functionality test failed: {e}")
        return False


def test_numpy_functionality():
    """Test basic NumPy functionality"""
    try:
        import numpy as np
        
        print("\n🔢 Testing NumPy functionality...")
        
        # Test array operations
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = arr1 + arr2
        print("   ✅ Basic array operations work")
        
        # Test random number generation
        np.random.seed(42)
        random_arr = np.random.randn(5, 5)
        print("   ✅ Random number generation works")
        
        # Test mathematical functions
        sin_vals = np.sin(arr1)
        print("   ✅ Mathematical functions work")
        
        return True
        
    except Exception as e:
        print(f"   ❌ NumPy functionality test failed: {e}")
        return False


def main():
    """Main test function"""
    print("🧪 Testing Python Environment for MNIST MLP Training")
    print("=" * 60)
    
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    # Test package imports and versions
    print("\n📦 Testing package installations...")
    
    tests_passed = []
    
    # Test required packages from requirements.txt
    tests_passed.append(test_import_and_version("torch", "1.9.0"))
    tests_passed.append(test_import_and_version("torchvision", "0.10.0"))
    tests_passed.append(test_import_and_version("numpy", "1.21.0"))
    
    # Test functionality
    tests_passed.append(test_torch_functionality())
    tests_passed.append(test_torchvision_functionality())
    tests_passed.append(test_numpy_functionality())
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    
    total_tests = len(tests_passed)
    passed_tests = sum(tests_passed)
    failed_tests = total_tests - passed_tests
    
    if all(tests_passed):
        print("🎉 All tests passed! Your environment is ready for MNIST MLP training.")
        print("\nYou can now run: python train.py")
        return 0
    else:
        print(f"❌ {failed_tests}/{total_tests} tests failed.")
        print("\nPlease install missing packages with: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    try:
        # Try to import packaging for version checking
        import packaging.version
    except ImportError:
        print("Installing packaging for version checking...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "packaging"])
        import packaging.version
    
    exit_code = main()
    sys.exit(exit_code)
