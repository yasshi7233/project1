import os
import argparse
import re
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Utilities
# ---------------------------
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

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100

# ---------------------------
# Feature preparation
# ---------------------------
def prepare_train_features(df, tfidf=None, tfidf_params=None, image_df=None, use_images=True):
    print("Preparing text and numeric features...")
    df = df.copy()
    df['cleaned_text'] = df['catalog_content'].apply(clean_text)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['text_length'] = df['cleaned_text'].str.len().fillna(0).astype(int)
    df['word_count'] = df['cleaned_text'].str.split().str.len().fillna(0).astype(int)

    numerical_features = ['ipq', 'text_length', 'word_count']
    X_num = df[numerical_features].astype(float).values

    # TF-IDF
    if tfidf is None:
        tfidf = TfidfVectorizer(
            max_features=(tfidf_params.get('max_features') if tfidf_params else 5000),
            ngram_range=(1,2),
            stop_words='english',
            min_df=5,
            max_df=0.95
        )
        X_text = tfidf.fit_transform(df['cleaned_text'].fillna(''))
    else:
        X_text = tfidf.transform(df['cleaned_text'].fillna(''))

    print(f"TF-IDF shape: {X_text.shape}")
    X_num_sparse = csr_matrix(X_num)

    X = hstack([X_text, X_num_sparse], format='csr')
    print(f"Text + numeric combined shape: {X.shape}")

    has_image = False
    if use_images and (image_df is not None):
        print("Merging image embeddings...")
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
                img_arr[pos] = image_df_indexed.loc[sid].values
            has_image = True
        else:
            print("No matching sample_id in image embeddings file.")
        X_img_sparse = csr_matrix(img_arr)
        X = hstack([X, X_img_sparse], format='csr')
        print(f"Combined with image features => shape: {X.shape}")
    else:
        print("Skipping image features (not provided or not requested).")

    y = np.log1p(df['price'].values.astype(float))
    return X, y, tfidf, numerical_features, has_image, df['sample_id'].astype(str).values

# ---------------------------
# Training
# ---------------------------
def train_and_save(train_csv, image_features_path, models_dir, use_images, test_size, random_state):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(train_csv)
    print(f"Loaded train: {df.shape}")

    # Load image features
    image_df = None
    if use_images and image_features_path and os.path.exists(image_features_path):
        print(f"Loading image features from {image_features_path} ...")
        image_df = pd.read_csv(image_features_path)
        print(f"Image features shape: {image_df.shape}")
    elif use_images:
        print("Image features file not found or path empty. Continuing without images.")
        image_df = None

    X, y, tfidf, numerical_features, has_image, sample_ids = prepare_train_features(
        df, tfidf=None, tfidf_params={'max_features':5000}, image_df=image_df, use_images=use_images
    )

    # Remove zero-variance features
    selector = VarianceThreshold(threshold=0.0)
    X = selector.fit_transform(X)
    print(f"After removing zero-variance features, X shape: {X.shape}")

    # Split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X, y, sample_ids, test_size=test_size, random_state=random_state
    )
    print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=10,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        feature_fraction=0.8
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=10)]
    )

    # Validation
    y_val_pred = lgb_model.predict(X_val)
    val_mae = mean_absolute_error(np.expm1(y_val), np.expm1(y_val_pred))
    val_smape = smape(np.expm1(y_val), np.expm1(y_val_pred))
    print(f"Validation MAE (price): {val_mae:.4f}, SMAPE: {val_smape:.4f}%")

    # Save artifacts
    joblib.dump(lgb_model, os.path.join(models_dir, 'final_lgb.pkl'))
    joblib.dump(tfidf, os.path.join(models_dir, 'final_tfidf.pkl'))
    feature_info = {
        'numerical_features': numerical_features,
        'has_image_features': has_image,
        'tfidf_max_features': 5000
    }
    joblib.dump(feature_info, os.path.join(models_dir, 'feature_info.pkl'))
    print(f"Saved model and artifacts to: {models_dir}")

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', default='data/train.csv')
    parser.add_argument('--image_features', default='project_data/embeddings/image_embeddings.csv')
    parser.add_argument('--models_dir', default='project_data/models')
    parser.add_argument('--use_images', action='store_true')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    train_and_save(args.train_data, args.image_features, args.models_dir, args.use_images, args.test_size, args.random_state)
