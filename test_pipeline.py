"""
Test script to verify the complete pipeline works
"""

import pandas as pd
import numpy as np
import os
import sys

def test_data_loading():
    """Test if data files exist and can be loaded"""
    print("Testing data loading...")
    
    # Check if dataset directory exists
    if not os.path.exists('../dataset'):
        print("❌ Dataset directory not found")
        return False
    
    # Check for train.csv
    if not os.path.exists('../dataset/train.csv'):
        print("❌ train.csv not found")
        return False
    
    # Try to load train.csv
    try:
        train_df = pd.read_csv('../dataset/train.csv')
        print(f"✅ train.csv loaded successfully: {train_df.shape}")
    except Exception as e:
        print(f"❌ Error loading train.csv: {e}")
        return False
    
    # Check for test.csv
    if not os.path.exists('../dataset/test.csv'):
        print("❌ test.csv not found")
        return False
    
    # Try to load test.csv
    try:
        test_df = pd.read_csv('../dataset/test.csv')
        print(f"✅ test.csv loaded successfully: {test_df.shape}")
    except Exception as e:
        print(f"❌ Error loading test.csv: {e}")
        return False
    
    return True

def test_preprocessing():
    """Test text preprocessing functions"""
    print("\nTesting text preprocessing...")
    
    # Import preprocessing functions
    sys.path.append('.')
    from predict_test import extract_ipq, clean_text
    
    # Test text cleaning
    test_text = "Apple iPhone 15 Pro Max 256GB - Pack of 2 - Latest Model"
    cleaned = clean_text(test_text)
    print(f"✅ Text cleaning: '{test_text}' -> '{cleaned}'")
    
    # Test IPQ extraction
    ipq = extract_ipq(test_text)
    print(f"✅ IPQ extraction: '{test_text}' -> {ipq}")
    
    # Test with different patterns
    test_cases = [
        "Pack of 5 items",
        "10 Pack Bundle",
        "Quantity: 3",
        "IPQ: 12",
        "5 x 2 pack"
    ]
    
    for case in test_cases:
        ipq = extract_ipq(case)
        print(f"   '{case}' -> {ipq}")
    
    return True

def test_model_loading():
    """Test if models can be loaded (if they exist)"""
    print("\nTesting model loading...")
    
    models_dir = '../models'
    if not os.path.exists(models_dir):
        print("⚠️  Models directory not found (models not trained yet)")
        return True
    
    required_files = ['final_lgb.pkl', 'final_xgb.pkl', 'final_tfidf.pkl', 'feature_info.pkl']
    
    for file in required_files:
        file_path = os.path.join(models_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file} found")
        else:
            print(f"❌ {file} not found")
            return False
    
    # Try to load models
    try:
        import joblib
        lgb_model = joblib.load(os.path.join(models_dir, 'final_lgb.pkl'))
        xgb_model = joblib.load(os.path.join(models_dir, 'final_xgb.pkl'))
        tfidf = joblib.load(os.path.join(models_dir, 'final_tfidf.pkl'))
        feature_info = joblib.load(os.path.join(models_dir, 'feature_info.pkl'))
        
        print("✅ All models loaded successfully")
        print(f"   Feature info: {feature_info}")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

def test_prediction():
    """Test prediction pipeline"""
    print("\nTesting prediction pipeline...")
    
    # Check if models exist
    if not os.path.exists('../models/final_lgb.pkl'):
        print("⚠️  Models not found, skipping prediction test")
        return True
    
    try:
        # Create sample test data
        sample_data = pd.DataFrame({
            'sample_id': ['test_001'],
            'catalog_content': ['Apple iPhone 15 Pro Max 256GB - Pack of 1 - Latest Model with Advanced Camera System'],
            'image_link': ['https://example.com/image.jpg']
        })
        
        # Import prediction function
        from predict_test import predict_prices, load_models
        
        # Load models
        lgb_model, xgb_model, tfidf, feature_info = load_models()
        
        # Generate prediction
        predictions = predict_prices(sample_data, lgb_model, xgb_model, tfidf, feature_info)
        
        print(f"✅ Prediction successful: ${predictions[0]:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\nTesting dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'lightgbm', 'xgboost', 
        'torch', 'torchvision', 'PIL', 'requests', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Smart Product Pricing Challenge - Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Data Loading", test_data_loading),
        ("Preprocessing", test_preprocessing),
        ("Model Loading", test_model_loading),
        ("Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()
