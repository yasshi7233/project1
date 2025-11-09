"""
Flask web application for testing product price predictions
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('.')

from predict_test import load_models, load_image_model, predict_prices

app = Flask(__name__)

# Global variables for models
models_loaded = False
lgb_model = None
xgb_model = None
tfidf = None
feature_info = None
image_model = None
image_transform = None
device = None

def load_all_models():
    """Load all models and preprocessing objects"""
    global models_loaded, lgb_model, xgb_model, tfidf, feature_info, image_model, image_transform, device
    
    try:
        print("Loading models...")
        lgb_model, xgb_model, tfidf, feature_info = load_models()
        
        if feature_info['has_image_features']:
            print("Loading image model...")
            image_model, image_transform, device = load_image_model()
        
        models_loaded = True
        print("All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict price for a single product"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        data = request.json
        catalog_content = data.get('catalog_content', '')
        image_link = data.get('image_link', '')
        
        # Create a temporary dataframe
        temp_df = pd.DataFrame({
            'sample_id': ['temp'],
            'catalog_content': [catalog_content],
            'image_link': [image_link]
        })
        
        # Generate prediction
        predictions = predict_prices(temp_df, lgb_model, xgb_model, tfidf, feature_info,
                                   image_model, image_transform, device)
        
        return jsonify({
            'predicted_price': float(predictions[0]),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict prices for multiple products"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded'}), 500
    
    try:
        # Get uploaded file
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read CSV file
        df = pd.read_csv(file)
        
        # Validate required columns
        required_columns = ['sample_id', 'catalog_content']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': f'Missing required columns: {required_columns}'}), 400
        
        # Generate predictions
        predictions = predict_prices(df, lgb_model, xgb_model, tfidf, feature_info,
                                   image_model, image_transform, device)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'sample_id': df['sample_id'],
            'price': predictions
        })
        
        # Save result
        output_path = 'submission/batch_predictions.csv'
        result_df.to_csv(output_path, index=False)
        
        return jsonify({
            'message': f'Predictions generated for {len(df)} products',
            'output_file': output_path,
            'sample_predictions': result_df.head().to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download prediction results"""
    try:
        return send_file(f'submission/{filename}', as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """Check model loading status"""
    return jsonify({
        'models_loaded': models_loaded,
        'has_image_features': feature_info['has_image_features'] if models_loaded else False
    })

if __name__ == '__main__':
    # Load models on startup
    if load_all_models():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check model files.")
