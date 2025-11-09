"""
Flask Web Application for ML Project - Product Price Prediction
Member 1: Flask test page for uploading test.csv and showing predicted prices
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
import sys

# Add src directory to path
sys.path.append('src')

from src.predict_test import TestPredictor
from src.data_pipeline import DataPipeline
from src.utils import FileManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_test_file(df):
    """Validate uploaded test file format"""
    required_columns = ['sample_id', 'catalog_content', 'image_link']
    
    if not all(col in df.columns for col in required_columns):
        return False, f"Missing required columns. Expected: {required_columns}"
    
    if df['sample_id'].isna().any():
        return False, "Found NaN values in sample_id column"
    
    if len(df) == 0:
        return False, "File is empty"
    
    return True, "File format is valid"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Load and validate file
            df = pd.read_csv(filepath)
            is_valid, message = validate_test_file(df)
            
            if not is_valid:
                flash(f'Invalid file format: {message}')
                os.remove(filepath)
                return redirect(url_for('index'))
            
            # Process file
            result = process_test_file(filepath, df)
            
            if result['success']:
                flash('Predictions generated successfully!')
                return render_template('results.html', 
                                     predictions=result['predictions'],
                                     summary=result['summary'],
                                     download_file=result['download_file'])
            else:
                flash(f'Error: {result["error"]}')
                return redirect(url_for('index'))
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a CSV file.')
        return redirect(url_for('index'))

def process_test_file(filepath, df):
    """Process uploaded test file and generate predictions"""
    try:
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Copy file to data directory
            temp_data_dir = os.path.join(temp_dir, 'data')
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # Save test file
            test_file_path = os.path.join(temp_data_dir, 'test.csv')
            df.to_csv(test_file_path, index=False)
            
            # Copy train.csv from main data directory to temp directory
            main_train_file = os.path.join('data', 'train.csv')
            if os.path.exists(main_train_file):
                import shutil
                temp_train_file = os.path.join(temp_data_dir, 'train.csv')
                shutil.copy2(main_train_file, temp_train_file)
                logger.info("Copied train.csv to temp directory")
            else:
                logger.warning("train.csv not found in main data directory")
            
            # Initialize pipeline and predictor
            pipeline = DataPipeline(temp_data_dir)
            predictor = TestPredictor()
            
            # Update predictor's file manager to use temp directory
            predictor.file_manager = FileManager(temp_dir)
            
            # Process data
            logger.info("Processing uploaded data...")
            _, test_df = pipeline.load_and_preprocess_data(test_file=test_file_path)
            X_test, _ = pipeline.prepare_features(pd.DataFrame(), test_df)
            
            # Load models (copy models to temp directory or use main directory)
            # First try to copy models to temp directory
            main_models_dir = 'models'
            temp_models_dir = os.path.join(temp_dir, 'models')
            if os.path.exists(main_models_dir):
                import shutil
                shutil.copytree(main_models_dir, temp_models_dir)
                logger.info("Copied models to temp directory")
            else:
                logger.warning("Models directory not found")
            
            # Load models
            predictor.load_models()
            
            # Generate predictions
            ensemble_pred, individual_preds = predictor.generate_ensemble_predictions(X_test)
            
            # Create results
            predictions_df = pd.DataFrame({
                'sample_id': test_df['sample_id'],
                'price': ensemble_pred
            })
            
            # Save predictions to temporary file
            output_file = os.path.join(temp_dir, 'predictions.csv')
            predictions_df.to_csv(output_file, index=False)
            
            # Create summary
            summary = {
                'total_samples': len(predictions_df),
                'price_range': f"${predictions_df['price'].min():.2f} - ${predictions_df['price'].max():.2f}",
                'mean_price': f"${predictions_df['price'].mean():.2f}",
                'median_price': f"${predictions_df['price'].median():.2f}",
                'models_used': list(individual_preds.keys())
            }
            
            return {
                'success': True,
                'predictions': predictions_df.head(20).to_dict('records'),  # Show first 20
                'summary': summary,
                'download_file': output_file
            }
            
    except Exception as e:
        logger.error(f"Error in process_test_file: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/download/<filename>')
def download_file(filename):
    """Download prediction results"""
    try:
        return send_file(filename, as_attachment=True, download_name='test_out.csv')
    except Exception as e:
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('index'))

@app.route('/sample')
def sample_data():
    """Show sample data format"""
    sample_data = {
        'sample_id': ['1', '2', '3'],
        'catalog_content': [
            'iPhone 15 Pro Max\nLatest smartphone with advanced features\nPack of 1',
            'Nike Air Max Shoes\nComfortable running shoes\nQuantity: 2 pairs',
            'Kitchen Knife Set\nProfessional chef knives\n5 pieces in set'
        ],
        'image_link': [
            'https://example.com/iphone.jpg',
            'https://example.com/shoes.jpg',
            'https://example.com/knives.jpg'
        ]
    }
    
    sample_df = pd.DataFrame(sample_data)
    return render_template('sample.html', sample_data=sample_df.to_dict('records'))

if __name__ == '__main__':
    # Check if models exist
    models_dir = 'models'
    if not os.path.exists(models_dir) or not os.listdir(models_dir):
        logger.warning("No trained models found. Please run baseline_model.py first.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
