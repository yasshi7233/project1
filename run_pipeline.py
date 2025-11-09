"""
Main Pipeline Runner Script for ML Project - Product Price Prediction
Member 1: Data pipeline, preprocessing, file management

This script runs the complete pipeline from data preprocessing to model training and prediction.
"""

import os
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.append('src')

from src.data_pipeline import DataPipeline
from src.baseline_model import BaselineModelTrainer
from src.predict_test import TestPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_data_files():
    """Check if required data files exist"""
    data_dir = Path("data")
    train_file = data_dir / "train.csv"
    test_file = data_dir / "test.csv"
    
    if not data_dir.exists():
        logger.error("Data directory not found. Creating data directory...")
        data_dir.mkdir(exist_ok=True)
        return False
    
    if not train_file.exists():
        logger.error(f"Training file not found: {train_file}")
        logger.info("Please place train.csv in the data/ directory")
        return False
    
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please place test.csv in the data/ directory")
        return False
    
    logger.info("All required data files found")
    return True

def run_data_pipeline():
    """Run data preprocessing pipeline"""
    logger.info("="*60)
    logger.info("STEP 1: DATA PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    try:
        pipeline = DataPipeline()
        results = pipeline.run_full_pipeline()
        
        logger.info("Data pipeline completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Data pipeline failed: {str(e)}")
        raise

def run_model_training():
    """Run baseline model training"""
    logger.info("="*60)
    logger.info("STEP 2: BASELINE MODEL TRAINING")
    logger.info("="*60)
    
    try:
        trainer = BaselineModelTrainer()
        models, results = trainer.train_all_models()
        
        logger.info("Model training completed successfully!")
        return models, results
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

def run_predictions():
    """Generate predictions for test data"""
    logger.info("="*60)
    logger.info("STEP 3: GENERATING TEST PREDICTIONS")
    logger.info("="*60)
    
    try:
        predictor = TestPredictor()
        submission, individual_preds = predictor.run_prediction_pipeline()
        
        logger.info("Prediction pipeline completed successfully!")
        return submission, individual_preds
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise

def main():
    """Main function to run complete pipeline"""
    logger.info("Starting ML Project Pipeline - Product Price Prediction")
    logger.info("Member 1: Data Pipeline, Preprocessing, File Management")
    
    try:
        # Check data files
        if not check_data_files():
            logger.error("Please ensure train.csv and test.csv are in the data/ directory")
            return
        
        # Step 1: Data Pipeline
        pipeline_results = run_data_pipeline()
        
        # Step 2: Model Training
        models, training_results = run_model_training()
        
        # Step 3: Generate Predictions
        submission, individual_preds = run_predictions()
        
        # Final Summary
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"Training samples processed: {len(pipeline_results['X_train'])}")
        print(f"Test samples processed: {len(pipeline_results['X_test'])}")
        print(f"Number of features: {len(pipeline_results['feature_names'])}")
        print(f"Models trained: {len(models)}")
        print(f"Predictions generated: {len(submission)}")
        
        # Best model
        best_model = min(training_results.items(), key=lambda x: x[1]['smape'])
        print(f"Best model: {best_model[0]} (SMAPE: {best_model[1]['smape']:.2f})")
        
        print(f"\nOutput files:")
        print(f"- test_out.csv: {os.path.join('output', 'test_out.csv')}")
        print(f"- Models: {os.path.join('models', '*.pkl')}")
        print(f"- Processed data: {os.path.join('data', '*_preprocessed.csv')}")
        
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
