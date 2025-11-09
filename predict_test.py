import pandas as pd
import numpy as np
import re
import joblib
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def extract_ipq(text):
    """Extract Item Pack Quantity from text"""
    patterns = [
        r'pack of (\d+)',
        r'(\d+) pack',
        r'quantity[\s:]+(\d+)',
        r'ipq[\s:]+(\d+)',
        r'(\d+)\s*x\s*(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            if len(match.groups()) == 2:
                return int(match.group(1)) * int(match.group(2))
            else:
                return int(match.group(1))
    return 1

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_image_features(image_path, model, transform, device):
    """Extract features using ResNet"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model(image_tensor)
            features = features.squeeze().cpu().numpy()
        
        return features
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_models():
    """Load trained models and preprocessing objects"""
    models_dir = '../models'
    
    # Load models
    lgb_model = joblib.load(os.path.join(models_dir, 'final_lgb.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'final_xgb.pkl'))
    tfidf = joblib.load(os.path.join(models_dir, 'final_tfidf.pkl'))
    feature_info = joblib.load(os.path.join(models_dir, 'feature_info.pkl'))
    
    return lgb_model, xgb_model, tfidf, feature_info

def load_image_model():
    """Load pretrained ResNet model for image feature extraction"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    resnet = models.resnet50(pretrained=True)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    resnet.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return resnet, transform, device

def predict_prices(test_df, lgb_model, xgb_model, tfidf, feature_info, image_model=None, image_transform=None, device=None):
    """Generate predictions for test data"""
    
    # Preprocess text
    test_df['cleaned_text'] = test_df['catalog_content'].apply(clean_text)
    test_df['ipq'] = test_df['catalog_content'].apply(extract_ipq)
    test_df['text_length'] = test_df['cleaned_text'].str.len()
    test_df['word_count'] = test_df['cleaned_text'].str.split().str.len()
    
    # Prepare text features
    tfidf_features = tfidf.transform(test_df['cleaned_text'])
    numerical_features = test_df[feature_info['numerical_features']].values
    X_text = hstack([tfidf_features, numerical_features])
    
    # Prepare image features if available
    if feature_info['has_image_features'] and image_model is not None:
        print("Extracting image features...")
        image_features = []
        
        for idx, row in test_df.iterrows():
            sample_id = row['sample_id']
            image_path = f"../images/{sample_id}.jpg"
            
            if os.path.exists(image_path):
                features = extract_image_features(image_path, image_model, image_transform, device)
                if features is not None:
                    image_features.append(features)
                else:
                    # Use zero features if extraction fails
                    image_features.append(np.zeros(2048))
            else:
                # Use zero features if image doesn't exist
                image_features.append(np.zeros(2048))
        
        image_features = np.array(image_features)
        X_combined = hstack([X_text, image_features])
    else:
        X_combined = X_text
    
    # Generate predictions
    print("Generating predictions...")
    lgb_pred = lgb_model.predict(X_combined)
    xgb_pred = xgb_model.predict(X_combined)
    
    # Ensemble prediction
    ensemble_pred = (lgb_pred + xgb_pred) / 2
    
    # Ensure positive predictions
    ensemble_pred = np.maximum(ensemble_pred, 0.01)
    
    return ensemble_pred

def main():
    """Main prediction function"""
    print("Loading test data...")
    test_df = pd.read_csv('../dataset/test.csv')
    print(f"Test data shape: {test_df.shape}")
    
    print("Loading models...")
    lgb_model, xgb_model, tfidf, feature_info = load_models()
    
    # Load image model if needed
    image_model, image_transform, device = None, None, None
    if feature_info['has_image_features']:
        print("Loading image model...")
        image_model, image_transform, device = load_image_model()
    
    print("Generating predictions...")
    predictions = predict_prices(test_df, lgb_model, xgb_model, tfidf, feature_info, 
                                image_model, image_transform, device)
    
    # Create submission file
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Save submission
    output_path = '../submission/test_out.csv'
    submission_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Submission shape: {submission_df.shape}")
    print(f"Price range: {predictions.min():.2f} - {predictions.max():.2f}")

if __name__ == "__main__":
    main()
