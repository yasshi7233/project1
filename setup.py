"""
Setup script for Smart Product Pricing Challenge
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'dataset',
        'models', 
        'images',
        'submission',
        'notebooks',
        'src/templates'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory already exists: {directory}")

def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling Python dependencies...")
    return run_command("pip install -r requirements.txt", "Installing requirements")

def download_sample_data():
    """Download sample data if not present"""
    print("\nChecking for sample data...")
    
    if not os.path.exists('dataset/train.csv'):
        print("⚠️  train.csv not found in dataset/ directory")
        print("Please download the dataset files and place them in the dataset/ directory")
        return False
    else:
        print("✅ train.csv found")
    
    if not os.path.exists('dataset/test.csv'):
        print("⚠️  test.csv not found in dataset/ directory")
        print("Please download the dataset files and place them in the dataset/ directory")
        return False
    else:
        print("✅ test.csv found")
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\nTesting installation...")
    return run_command("python src/test_pipeline.py", "Running pipeline test")

def main():
    """Main setup function"""
    print("Smart Product Pricing Challenge - Setup")
    print("=" * 40)
    
    # Create directories
    print("\n1. Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n2. Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Check for data
    print("\n3. Checking for data files...")
    if not download_sample_data():
        print("⚠️  Data files not found. Please download and place them in dataset/ directory")
        print("You can still proceed with the setup, but you'll need the data files to train models")
    
    # Test installation
    print("\n4. Testing installation...")
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Place train.csv and test.csv in the dataset/ directory")
        print("2. Run: python src/train_model.py --use_images")
        print("3. Run: python src/predict_test.py")
        print("4. Or use Jupyter notebooks for interactive development")
    else:
        print("\n⚠️  Setup completed with warnings. Please check the test output above.")
    
    return True

if __name__ == "__main__":
    main()
