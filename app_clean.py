"""
Clean Flask Web Application for ML Project - Product Price Prediction
No dependencies on complex preprocessing - just simple predictions
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import pandas as pd
import numpy as np
import os
import logging
from werkzeug.utils import secure_filename
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

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

def extract_quantity(text):
    """Extract quantity from text"""
    if pd.isna(text):
        return 1
    
    text = str(text).lower()
    
    # Look for quantity patterns
    patterns = [
        r'(\d+)\s*(?:pack|pcs|pieces|units|items|count)',
        r'(\d+)\s*(?:x|×)',
        r'pack\s*of\s*(\d+)',
        r'(\d+)\s*(?:per\s*pack|per\s*box)',
        r'quantity[:\s]*(\d+)',
        r'qty[:\s]*(\d+)',
        r'(\d+)\s*(?:ct|count)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return int(match.group(1))
            except:
                continue
    
    return 1

def generate_smart_predictions(df):
    """Generate smart predictions based on text analysis"""
    logger.info("Generating smart predictions...")
    
    predictions = []
    
    for idx, row in df.iterrows():
        try:
            # Get text content
            text = str(row['catalog_content']).lower()
            
            # Base price
            base_price = 15.0
            
            # 1. Text length factor
            text_length = len(text)
            if text_length > 1000:
                base_price *= 3.0
            elif text_length > 500:
                base_price *= 2.2
            elif text_length > 200:
                base_price *= 1.5
            elif text_length > 100:
                base_price *= 1.2
            
            # 2. Premium keywords
            premium_keywords = [
                'premium', 'professional', 'advanced', 'ultra', 'pro', 'max', 
                'high-end', 'luxury', 'deluxe', 'premium quality', 'top quality'
            ]
            premium_count = sum(1 for keyword in premium_keywords if keyword in text)
            if premium_count > 0:
                base_price *= (1 + premium_count * 0.3)
            
            # 3. Electronics category
            electronics_keywords = [
                'electronic', 'digital', 'smart', 'wireless', 'bluetooth', 
                'battery', 'charger', 'cable', 'wire', 'circuit', 'chip',
                'processor', 'memory', 'storage', 'display', 'screen'
            ]
            if any(keyword in text for keyword in electronics_keywords):
                base_price *= 2.5
            
            # 4. Clothing/Fashion category
            clothing_keywords = [
                'shirt', 'dress', 'pants', 'jacket', 'shoes', 'clothing',
                'fashion', 'style', 'wear', 'apparel', 'garment', 'fabric'
            ]
            if any(keyword in text for keyword in clothing_keywords):
                base_price *= 1.8
            
            # 5. Home/Kitchen category
            home_keywords = [
                'home', 'kitchen', 'bedroom', 'bathroom', 'furniture',
                'cookware', 'utensil', 'appliance', 'decor', 'furnishing'
            ]
            if any(keyword in text for keyword in home_keywords):
                base_price *= 1.6
            
            # 6. Beauty/Health category
            beauty_keywords = [
                'beauty', 'cosmetic', 'makeup', 'skincare', 'perfume',
                'health', 'vitamin', 'supplement', 'medicine', 'care'
            ]
            if any(keyword in text for keyword in beauty_keywords):
                base_price *= 2.0
            
            # 7. Sports/Fitness category
            sports_keywords = [
                'sports', 'fitness', 'gym', 'exercise', 'outdoor',
                'athletic', 'training', 'workout', 'equipment'
            ]
            if any(keyword in text for keyword in sports_keywords):
                base_price *= 1.7
            
            # 8. Quantity factor
            quantity = extract_quantity(text)
            if quantity > 20:
                base_price *= 1.8
            elif quantity > 10:
                base_price *= 1.5
            elif quantity > 5:
                base_price *= 1.3
            elif quantity > 1:
                base_price *= 1.1
            
            # 9. Brand indicators
            brand_indicators = [
                'brand', 'manufacturer', 'original', 'authentic', 'genuine',
                'official', 'licensed', 'certified', 'warranty', 'guarantee'
            ]
            if any(indicator in text for indicator in brand_indicators):
                base_price *= 1.4
            
            # 10. Size indicators
            size_indicators = [
                'large', 'big', 'huge', 'giant', 'jumbo', 'mega',
                'small', 'mini', 'tiny', 'compact', 'portable'
            ]
            if any(indicator in text for indicator in size_indicators):
                if any(word in text for word in ['large', 'big', 'huge', 'giant', 'jumbo', 'mega']):
                    base_price *= 1.3
                else:
                    base_price *= 0.8
            
            # Add some randomness (5% variation)
            base_price *= np.random.uniform(0.95, 1.05)
            
            # Ensure reasonable range
            base_price = max(0.5, min(base_price, 2000.0))
            
            predictions.append(round(base_price, 2))
            
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {str(e)}")
            predictions.append(25.0)  # Default price
    
    return predictions

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
        logger.info(f"Processing {len(df)} samples...")
        
        # Generate smart predictions
        predictions = generate_smart_predictions(df)
        
        # Create results
        predictions_df = pd.DataFrame({
            'sample_id': df['sample_id'],
            'price': predictions
        })
        
        # Save predictions to file
        output_file = os.path.join(app.config['UPLOAD_FOLDER'], 'predictions.csv')
        predictions_df.to_csv(output_file, index=False)
        
        # Create summary
        summary = {
            'total_samples': len(predictions_df),
            'price_range': f"${predictions_df['price'].min():.2f} - ${predictions_df['price'].max():.2f}",
            'mean_price': f"${predictions_df['price'].mean():.2f}",
            'median_price': f"${predictions_df['price'].median():.2f}",
            'models_used': ['Smart Heuristic Model']
        }
        
        return {
            'success': True,
            'predictions': predictions_df.head(20).to_dict('records'),
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        return send_file(filepath, as_attachment=True, download_name='test_out.csv')
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
    app.run(debug=True, host='0.0.0.0', port=5000)
