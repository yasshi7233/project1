"""
Setup script to create necessary directories for ML Project
Member 1: Data pipeline, preprocessing, file management
"""

import os
from pathlib import Path

def create_directories():
    """Create all necessary directories for the project"""
    
    directories = [
        "data",
        "models", 
        "output",
        "uploads",
        "templates",
        "src"
    ]
    
    print("Creating project directories...")
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    print("\nProject structure created successfully!")
    print("\nNext steps:")
    print("1. Place train.csv and test.csv in the data/ directory")
    print("2. Run: pip install -r requirements.txt")
    print("3. Run: python run_pipeline.py")

if __name__ == "__main__":
    create_directories()
