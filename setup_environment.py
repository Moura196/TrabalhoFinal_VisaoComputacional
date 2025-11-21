"""
Environment setup script for the Computer Vision project.
This script helps users set up their development environment.
"""

import sys
import subprocess
import os
from pathlib import Path


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60 + "\n")


def check_python_version():
    """Check if Python version is compatible."""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def install_requirements():
    """Install required packages from requirements.txt."""
    print_header("INSTALLING REQUIRED PACKAGES")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    print("Installing packages from requirements.txt...")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("\n✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing packages: {e}")
        return False


def run_verification():
    """Run the verification script."""
    print_header("RUNNING ENVIRONMENT VERIFICATION")
    
    verify_script = Path(__file__).parent / "verify_environment.py"
    
    if not verify_script.exists():
        print("✗ verify_environment.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(verify_script)])
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error running verification: {e}")
        return False


def test_preprocessing_pipeline():
    """Test the data preprocessing pipeline."""
    print_header("TESTING DATA PREPROCESSING PIPELINE")
    
    preprocessing_script = Path(__file__).parent / "data_preprocessing.py"
    
    if not preprocessing_script.exists():
        print("✗ data_preprocessing.py not found")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(preprocessing_script)])
        return result.returncode == 0
    except Exception as e:
        print(f"✗ Error testing preprocessing pipeline: {e}")
        return False


def main():
    """Main setup function."""
    print("\n" + "=" * 60)
    print("COMPUTER VISION PROJECT - ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        print("\nPlease install Python 3.8 or higher and try again.")
        return 1
    
    # Ask user if they want to install packages
    print("\nDo you want to install required packages? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        if not install_requirements():
            print("\nPackage installation failed.")
            print("Please check the error messages and try again.")
            return 1
    else:
        print("\nSkipping package installation.")
        print("Make sure to install packages manually using:")
        print("  pip install -r requirements.txt")
    
    # Run verification
    print("\nDo you want to run environment verification? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        if not run_verification():
            print("\nEnvironment verification failed.")
            print("Please check the error messages and fix any issues.")
            return 1
    
    # Test preprocessing pipeline
    print("\nDo you want to test the data preprocessing pipeline? (y/n): ", end="")
    response = input().strip().lower()
    
    if response == 'y' or response == 'yes':
        if not test_preprocessing_pipeline():
            print("\nData preprocessing pipeline test failed.")
            print("Please check the error messages and fix any issues.")
            return 1
    
    # Final message
    print_header("SETUP COMPLETE")
    print("✓ Environment setup completed successfully!")
    print("\nYou can now start working on your computer vision project.")
    print("\nQuick start:")
    print("  1. Import preprocessing: from data_preprocessing import get_preprocessing_pipeline")
    print("  2. Create preprocessor: preprocessor = get_preprocessing_pipeline(mode='train')")
    print("  3. Load and preprocess images: tensor = preprocessor(image)")
    print("\nFor more information, check the documentation in each module.\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
