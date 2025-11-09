"""
Baseline Model Training Script for ML Project - Product Price Prediction
Member 1: Data pipeline, preprocessing, file management
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import pickle
import json
from utils import calculate_smape, log_transform_prices, inverse_log_transform, FileManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModelTrainer:
    """Class to train baseline models"""
    
    def __init__(self):
        self.file_manager = FileManager()
        self.models = {}
        self.results = {}
    
    def load_processed_data(self):
        """Load preprocessed data from pipeline"""
        logger.info("Loading processed data...")
        
        # Load features
        X_train = pd.read_csv(os.path.join(self.file_manager.data_dir, "features_train.csv"))
        X_test = pd.read_csv(os.path.join(self.file_manager.data_dir, "features_test.csv"))
        
        # Load targets
        y_train = pd.read_csv(os.path.join(self.file_manager.data_dir, "y_train.csv"))['price'].values
        y_val = pd.read_csv(os.path.join(self.file_manager.data_dir, "y_val.csv"))['price'].values
        
        # Split train data back to train/val
        val_size = len(y_val)
        X_train_split = X_train.iloc[:-val_size]
        X_val_split = X_train.iloc[-val_size:]
        y_train_split = y_train[:-val_size]
        
        logger.info(f"Loaded data - Train: {X_train_split.shape}, Val: {X_val_split.shape}, Test: {X_test.shape}")
        
        return X_train_split, X_val_split, X_test, y_train_split, y_val
    
    def train_linear_regression(self, X_train, y_train, X_val, y_val):
        """Train Linear Regression model"""
        logger.info("Training Linear Regression...")
        
        # Apply log transformation
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        model = LinearRegression()
        model.fit(X_train, y_train_log)
        
        # Predictions
        y_pred_log = model.predict(X_val)
        y_pred = inverse_log_transform(y_pred_log)
        
        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        smape = calculate_smape(y_val, y_pred)
        
        self.results['linear_regression'] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'smape': smape
        }
        
        self.models['linear_regression'] = model
        
        logger.info(f"Linear Regression - MAE: {mae:.2f}, SMAPE: {smape:.2f}")
        return model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        # Apply log transformation
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train_log)
        
        # Predictions
        y_pred_log = model.predict(X_val)
        y_pred = inverse_log_transform(y_pred_log)
        
        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        smape = calculate_smape(y_val, y_pred)
        
        self.results['random_forest'] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'smape': smape
        }
        
        self.models['random_forest'] = model
        
        logger.info(f"Random Forest - MAE: {mae:.2f}, SMAPE: {smape:.2f}")
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")
        
        # Apply log transformation
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train_log)
        val_data = lgb.Dataset(X_val, label=y_val_log, reference=train_data)
        
        # Parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )
        
        # Predictions
        y_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = inverse_log_transform(y_pred_log)
        
        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        smape = calculate_smape(y_val, y_pred)
        
        self.results['lightgbm'] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'smape': smape
        }
        
        self.models['lightgbm'] = model
        
        logger.info(f"LightGBM - MAE: {mae:.2f}, SMAPE: {smape:.2f}")
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        
        # Apply log transformation
        y_train_log = np.log1p(y_train)
        y_val_log = np.log1p(y_val)
        
        model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=100,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train_log,
            eval_set=[(X_val, y_val_log)],
            verbose=False
        )
        
        # Predictions
        y_pred_log = model.predict(X_val)
        y_pred = inverse_log_transform(y_pred_log)
        
        # Metrics
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        smape = calculate_smape(y_val, y_pred)
        
        self.results['xgboost'] = {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'smape': smape
        }
        
        self.models['xgboost'] = model
        
        logger.info(f"XGBoost - MAE: {mae:.2f}, SMAPE: {smape:.2f}")
        return model
    
    def train_all_models(self):
        """Train all baseline models"""
        logger.info("Starting baseline model training...")
        
        # Load data
        X_train, X_val, X_test, y_train, y_val = self.load_processed_data()
        
        # Train models
        self.train_linear_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        
        # Save models
        self.save_models()
        
        # Print results summary
        self.print_results_summary()
        
        return self.models, self.results
    
    def save_models(self):
        """Save all trained models"""
        logger.info("Saving models...")
        
        for model_name, model in self.models.items():
            if model_name == 'lightgbm':
                # LightGBM model
                model.save_model(os.path.join(self.file_manager.models_dir, f'{model_name}.txt'))
            else:
                # Other models
                with open(os.path.join(self.file_manager.models_dir, f'{model_name}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
        
        # Save results
        with open(os.path.join(self.file_manager.models_dir, 'model_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info("Models saved successfully")
    
    def print_results_summary(self):
        """Print results summary"""
        print("\n" + "="*60)
        print("BASELINE MODEL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'MAE':<10} {'MSE':<12} {'R²':<8} {'SMAPE':<8}")
        print("-"*60)
        
        for model_name, results in self.results.items():
            print(f"{model_name:<20} {results['mae']:<10.2f} {results['mse']:<12.2f} "
                  f"{results['r2']:<8.3f} {results['smape']:<8.2f}")
        
        # Find best model
        best_model = min(self.results.items(), key=lambda x: x[1]['smape'])
        print("-"*60)
        print(f"Best Model (by SMAPE): {best_model[0]} - SMAPE: {best_model[1]['smape']:.2f}")
        print("="*60)

def main():
    """Main function to train baseline models"""
    trainer = BaselineModelTrainer()
    
    try:
        models, results = trainer.train_all_models()
        logger.info("Baseline model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
