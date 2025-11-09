"""
Data Pipeline Script for ML Project - Product Price Prediction
Member 1: Data pipeline, preprocessing, file management
"""

import pandas as pd
import numpy as np
import os
import logging
from utils import DataPreprocessor, FileManager, ImageDownloader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import pickle

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPipeline:
    """Main data pipeline class"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.preprocessor = DataPreprocessor()
        self.file_manager = FileManager()
        self.image_downloader = ImageDownloader()
        
        # Initialize feature extractors
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.scaler = StandardScaler()
    
    def load_and_preprocess_data(self, train_file: str = "train.csv", test_file: str = "test.csv"):
        """Load and preprocess both train and test data"""
        logger.info("Starting data loading and preprocessing...")
        
        # Load data
        train_df = self.preprocessor.load_data(os.path.join(self.data_dir, train_file))
        test_df = self.preprocessor.load_data(os.path.join(self.data_dir, test_file))
        
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        
        # Preprocess catalog content
        train_df = self.preprocessor.preprocess_catalog_content(train_df)
        test_df = self.preprocessor.preprocess_catalog_content(test_df)
        
        # Create text features
        train_df = self.preprocessor.create_text_features(train_df)
        test_df = self.preprocessor.create_text_features(test_df)
        
        # Save preprocessed data
        self.file_manager.save_preprocessed_data(train_df, "train_preprocessed.csv")
        self.file_manager.save_preprocessed_data(test_df, "test_preprocessed.csv")
        
        logger.info("Data preprocessing completed")
        return train_df, test_df
    
    def create_text_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Create TF-IDF features from text data"""
        logger.info("Creating TF-IDF features...")
        
        # Fit TF-IDF on training data
        train_text = train_df['combined_text'].fillna('')
        test_text = test_df['combined_text'].fillna('')
        
        train_tfidf = self.tfidf_vectorizer.fit_transform(train_text)
        test_tfidf = self.tfidf_vectorizer.transform(test_text)
        
        # Convert to DataFrame
        feature_names = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(
            train_tfidf.toarray(),
            columns=feature_names,
            index=train_df.index
        )
        test_tfidf_df = pd.DataFrame(
            test_tfidf.toarray(),
            columns=feature_names,
            index=test_df.index
        )
        
        # Combine with original features
        train_features = pd.concat([train_df, train_tfidf_df], axis=1)
        test_features = pd.concat([test_df, test_tfidf_df], axis=1)
        
        logger.info(f"Created {len(feature_names)} TF-IDF features")
        return train_features, test_features
    
    def prepare_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Prepare final feature set for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Select numerical features
        numerical_features = [
            'quantity', 'title_length', 'description_length', 'total_text_length',
            'title_word_count', 'description_word_count', 'total_word_count',
            'is_electronics', 'is_clothing', 'is_home', 'is_beauty', 'is_sports'
        ]
        
        # Create TF-IDF features
        train_features, test_features = self.create_text_features(train_df, test_df)
        
        # Get TF-IDF feature names
        tfidf_features = [col for col in train_features.columns if col.startswith('tfidf_')]
        
        # Combine all features
        all_features = numerical_features + tfidf_features
        
        # Prepare feature matrices
        X_train = train_features[all_features].fillna(0)
        X_test = test_features[all_features].fillna(0)
        
        # Scale numerical features
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        # Scale only numerical features (not TF-IDF)
        X_train_scaled[numerical_features] = self.scaler.fit_transform(X_train[numerical_features])
        X_test_scaled[numerical_features] = self.scaler.transform(X_test[numerical_features])
        
        # Get target variable
        y_train = train_df['price'].values
        
        logger.info(f"Final feature matrix shape - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, all_features
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2):
        """Split data for validation"""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Validation: {X_val.shape}")
        return X_train, X_val, y_train, y_val
    
    def save_feature_info(self, feature_names: list):
        """Save feature information for later use"""
        feature_info = {
            'feature_names': feature_names,
            'num_features': len(feature_names)
        }
        
        with open(os.path.join(self.file_manager.models_dir, 'feature_info.json'), 'w') as f:
            import json
            json.dump(feature_info, f)
        
        logger.info(f"Feature info saved: {len(feature_names)} features")
    
    def run_full_pipeline(self, train_file: str = "train.csv", test_file: str = "test.csv"):
        """Run the complete data pipeline"""
        logger.info("Starting full data pipeline...")
        
        # Load and preprocess data
        train_df, test_df = self.load_and_preprocess_data(train_file, test_file)
        
        # Prepare features
        X_train, X_test, y_train, feature_names = self.prepare_features(train_df, test_df)
        
        # Save feature information
        self.save_feature_info(feature_names)
        
        # Split data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = self.split_data(X_train, y_train)
        
        # Save processed data
        self.file_manager.save_preprocessed_data(
            pd.concat([X_train_split, X_val_split], axis=0), 
            "features_train.csv"
        )
        self.file_manager.save_preprocessed_data(X_test, "features_test.csv")
        
        # Save target variables
        pd.DataFrame({'price': y_train_split}).to_csv(
            os.path.join(self.file_manager.data_dir, "y_train.csv"), index=False
        )
        pd.DataFrame({'price': y_val_split}).to_csv(
            os.path.join(self.file_manager.data_dir, "y_val.csv"), index=False
        )
        
        logger.info("Full data pipeline completed successfully")
        
        return {
            'X_train': X_train_split,
            'X_val': X_val_split,
            'X_test': X_test,
            'y_train': y_train_split,
            'y_val': y_val_split,
            'feature_names': feature_names,
            'train_df': train_df,
            'test_df': test_df
        }

def main():
    """Main function to run data pipeline"""
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Check if data files exist
    train_file = os.path.join("data", "train.csv")
    test_file = os.path.join("data", "test.csv")
    
    if not os.path.exists(train_file):
        logger.error(f"Training file not found: {train_file}")
        logger.info("Please place train.csv in the data/ directory")
        return
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please place test.csv in the data/ directory")
        return
    
    # Run pipeline
    try:
        results = pipeline.run_full_pipeline()
        logger.info("Data pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("DATA PIPELINE SUMMARY")
        print("="*50)
        print(f"Training samples: {len(results['X_train'])}")
        print(f"Validation samples: {len(results['X_val'])}")
        print(f"Test samples: {len(results['X_test'])}")
        print(f"Number of features: {len(results['feature_names'])}")
        print(f"Price range: ${results['y_train'].min():.2f} - ${results['y_train'].max():.2f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
