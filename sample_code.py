"""
Sample code for Smart Product Pricing Challenge
This is a dummy implementation that generates random predictions
"""

import pandas as pd
import numpy as np

def generate_sample_predictions():
    """Generate sample predictions for test data"""
    
    # Load test data
    test_df = pd.read_csv('dataset/test.csv')
    print(f"Loaded test data with {len(test_df)} samples")
    
    # Generate random predictions (replace with your actual model)
    np.random.seed(42)
    predictions = np.random.uniform(10, 1000, len(test_df))
    
    # Create submission dataframe
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save submission
    submission_df.to_csv('submission/test_out.csv', index=False)
    print(f"Sample predictions saved to submission/test_out.csv")
    print(f"Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")

if __name__ == "__main__":
    generate_sample_predictions()
