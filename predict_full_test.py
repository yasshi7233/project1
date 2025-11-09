#!/usr/bin/env python3
"""
Load trained LightGBM + TF-IDF + feature_info and predict prices for test set.

Usage:
python src/predict_full_test.py \
  --test_data data/test.csv \
  --image_features project_data/embeddings/image_embeddings.csv \
  --models_dir project_data/models \
  --output project_data/submission/test_out.csv \
  --use_images
"""
import os
import argparse
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# helper copied from train
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_ipq(text):
    if pd.isna(text):
        return 1
    text = str(text).lower()
    patterns = [
        r'pack of (\d+)',
        r'(\d+)\s*pack',
        r'quantity[\s:]+(\d+)',
        r'ipq[\s:]+(\d+)',
        r'(\d+)\s*x\s*(\d+)'
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            if len(m.groups()) == 2:
                try:
                    return int(m.group(1)) * int(m.group(2))
                except:
                    return 1
            else:
                try:
                    return int(m.group(1))
                except:
                    return 1
    return 1

def prepare_test_features(df, tfidf, numerical_features, image_df=None, use_images=True):
    df = df.copy()
    df['cleaned_text'] = df['catalog_content'].apply(clean_text)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['text_length'] = df['cleaned_text'].str.len().fillna(0).astype(int)
    df['word_count'] = df['cleaned_text'].str.split().str.len().fillna(0).astype(int)

    X_num = df[numerical_features].astype(float).values
    X_num_sparse = csr_matrix(X_num)

    X_text = tfidf.transform(df['cleaned_text'].fillna(''))
    X = hstack([X_text, X_num_sparse], format='csr')

    if use_images and (image_df is not None):
        # align image_df to df
        image_df_indexed = image_df.copy()
        if 'sample_id' in image_df_indexed.columns:
            image_df_indexed = image_df_indexed.set_index('sample_id')
        image_df_indexed.index = image_df_indexed.index.astype(str)
        ids = df['sample_id'].astype(str).values
        img_feat_dim = image_df_indexed.shape[1]
        img_arr = np.zeros((len(ids), img_feat_dim), dtype=np.float32)
        present = image_df_indexed.index.intersection(ids)
        if len(present) > 0:
            id_to_pos = {sid: i for i, sid in enumerate(ids)}
            for sid in present:
                pos = id_to_pos[sid]
                try:
                    img_arr[pos] = image_df_indexed.loc[sid].values
                except Exception:
                    pass
        X_img_sparse = csr_matrix(img_arr)
        X = hstack([X, X_img_sparse], format='csr')
    return X

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', default='data/test.csv')
    parser.add_argument('--image_features', default='project_data/embeddings/image_embeddings.csv')
    parser.add_argument('--models_dir', default='project_data/models')
    parser.add_argument('--output', default='project_data/submission/test_out.csv')
    parser.add_argument('--use_images', action='store_true')
    args = parser.parse_args()

    # load artifacts
    model_path = os.path.join(args.models_dir, 'final_lgb.pkl')
    tfidf_path = os.path.join(args.models_dir, 'final_tfidf.pkl')
    info_path = os.path.join(args.models_dir, 'feature_info.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"TF-IDF not found: {tfidf_path}")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"feature_info not found: {info_path}")

    print("Loading model and preprocessing artifacts...")
    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    feature_info = joblib.load(info_path)
    numerical_features = feature_info['numerical_features']

    # load test data
    df_test = pd.read_csv(args.test_data)
    print(f"Loaded test: {df_test.shape}")

    image_df = None
    if args.use_images and os.path.exists(args.image_features):
        print(f"Loading image features from {args.image_features} ...")
        image_df = pd.read_csv(args.image_features)
        print(f"Image features: {image_df.shape}")
    elif args.use_images:
        print("Image features requested but file not found. Proceeding without image features.")
        image_df = None

    X_test = prepare_test_features(df_test, tfidf, numerical_features, image_df=image_df, use_images=args.use_images)
    print(f"Prepared test features shape: {X_test.shape}")

    print("Predicting...")
    y_pred_log = model.predict(X_test)
    # invert log1p
    y_pred = np.expm1(y_pred_log)
    # ensure no negative or zero predictions
    y_pred = np.where(y_pred <= 0, 1.0, y_pred)

    out_df = pd.DataFrame({
        'sample_id': df_test['sample_id'].astype(str),
        'price': y_pred
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")
