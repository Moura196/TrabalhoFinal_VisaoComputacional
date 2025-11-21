"""
Verification script to test environment setup and GPU/CPU availability.
This script validates that all required dependencies are properly installed
and checks if PyTorch can utilize GPU acceleration.
"""

import sys
from typing import Dict, List, Tuple


def check_imports() -> Tuple[bool, List[str]]:
    """
    Test importing all required packages.
    
    Returns:
        Tuple of (success status, list of messages)
    """
    messages = []
    all_success = True
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'Torchvision'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('PIL', 'Pillow'),
    ]
    
    print("=" * 60)
    print("TESTING PACKAGE IMPORTS")
    print("=" * 60)
    
    for package_name, display_name in required_packages:
        try:
            __import__(package_name)
            messages.append(f"✓ {display_name} imported successfully")
            print(f"✓ {display_name} imported successfully")
        except ImportError as e:
            messages.append(f"✗ {display_name} import failed: {e}")
            print(f"✗ {display_name} import failed: {e}")
            all_success = False
    
    print()
    return all_success, messages


def check_pytorch_gpu() -> Tuple[bool, List[str]]:
    """
    Check PyTorch GPU/CPU availability and configuration.
    
    Returns:
        Tuple of (success status, list of messages)
    """
    messages = []
    
    try:
        import torch
        
        print("=" * 60)
        print("PYTORCH CONFIGURATION")
        print("=" * 60)
        
        # PyTorch version
        pytorch_version = torch.__version__
        messages.append(f"PyTorch version: {pytorch_version}")
        print(f"PyTorch version: {pytorch_version}")
        
        # CUDA availability
        cuda_available = torch.cuda.is_available()
        messages.append(f"CUDA available: {cuda_available}")
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            # CUDA version
            cuda_version = torch.version.cuda
            messages.append(f"CUDA version: {cuda_version}")
            print(f"CUDA version: {cuda_version}")
            
            # Number of GPUs
            gpu_count = torch.cuda.device_count()
            messages.append(f"Number of GPUs: {gpu_count}")
            print(f"Number of GPUs: {gpu_count}")
            
            # Current GPU
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            messages.append(f"Current GPU: {gpu_name}")
            print(f"Current GPU: {gpu_name}")
        else:
            messages.append("Running on CPU only")
            print("Running on CPU only")
        
        print()
        return True, messages
        
    except Exception as e:
        messages.append(f"Error checking PyTorch configuration: {e}")
        print(f"Error checking PyTorch configuration: {e}")
        print()
        return False, messages


def test_tensor_operations() -> Tuple[bool, List[str]]:
    """
    Test basic tensor operations on available device (GPU/CPU).
    
    Returns:
        Tuple of (success status, list of messages)
    """
    messages = []
    
    try:
        import torch
        
        print("=" * 60)
        print("TESTING TENSOR OPERATIONS")
        print("=" * 60)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        messages.append(f"Using device: {device}")
        print(f"Using device: {device}")
        
        # Create test tensors
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        
        # Perform operations
        z = torch.matmul(x, y)
        
        messages.append(f"Matrix multiplication test: SUCCESS")
        print(f"Matrix multiplication test: SUCCESS")
        print(f"Result shape: {z.shape}")
        print(f"Result sample:\n{z[:2, :2]}")
        
        # Test moving between CPU and GPU if CUDA is available
        if torch.cuda.is_available():
            cpu_tensor = z.cpu()
            gpu_tensor = cpu_tensor.cuda()
            messages.append("CPU ↔ GPU transfer test: SUCCESS")
            print("CPU ↔ GPU transfer test: SUCCESS")
        
        print()
        return True, messages
        
    except Exception as e:
        messages.append(f"Error in tensor operations: {e}")
        print(f"Error in tensor operations: {e}")
        print()
        return False, messages


def hello_world_training() -> Tuple[bool, List[str]]:
    """
    Run a simple 'hello world' training example to verify the setup.
    
    Returns:
        Tuple of (success status, list of messages)
    """
    messages = []
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        print("=" * 60)
        print("HELLO WORLD TRAINING TEST")
        print("=" * 60)
        
        # Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        messages.append(f"Training on device: {device}")
        print(f"Training on device: {device}")
        
        # Simple linear model
        model = nn.Linear(10, 1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Dummy data
        x_train = torch.randn(100, 10, device=device)
        y_train = torch.randn(100, 1, device=device)
        
        # Training loop
        model.train()
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            if epoch == 0 or epoch == 4:
                print(f"Epoch {epoch + 1}/5, Loss: {loss.item():.4f}")
        
        messages.append("Training test completed successfully")
        print("Training test completed successfully")
        print()
        return True, messages
        
    except Exception as e:
        messages.append(f"Error in training test: {e}")
        print(f"Error in training test: {e}")
        print()
        return False, messages


def main():
    """
    Main function to run all verification tests.
    """
    print("\n" + "=" * 60)
    print("ENVIRONMENT VERIFICATION SCRIPT")
    print("=" * 60)
    print()
    
    results = []
    
    # Run all checks
    results.append(("Package Imports", check_imports()))
    results.append(("PyTorch Configuration", check_pytorch_gpu()))
    results.append(("Tensor Operations", test_tensor_operations()))
    results.append(("Training Test", hello_world_training()))
    
    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, (success, _) in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All verification tests passed!")
        print("Your environment is ready for training.\n")
        return 0
    else:
        print("\n✗ Some verification tests failed.")
        print("Please check the error messages above and fix the issues.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
