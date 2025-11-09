
import os
import io
import time
import requests
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm
import argparse


# -----------------------------
# IMAGE DOWNLOAD
# -----------------------------
def _safe_fetch_image(url, timeout=30):
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert('RGB')
        return img
    except Exception:
        return None

def download_images(df, image_dir='images', url_col='image_link', id_col='sample_id', num_workers=16, max_retries=3, delay=1, overwrite=False):
    """
    Download images from URLs in parallel

    Returns:
        List of failed sample_ids
    """
    os.makedirs(image_dir, exist_ok=True)
    pairs = list(df[[id_col, url_col]].itertuples(index=False, name=None))

    def _save(pair):
        sid, url = pair
        fname = os.path.join(image_dir, f"{sid}.jpg")
        if os.path.exists(fname) and not overwrite:
            return sid, True
        for attempt in range(max_retries):
            img = _safe_fetch_image(url)
            if img is not None:
                try:
                    img.save(fname, format='JPEG', quality=85)
                    return sid, True
                except:
                    pass
            time.sleep(delay * (attempt+1))
        return sid, False

    failed = []
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(_save, p): p for p in pairs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Downloading images'):
            sid, ok = fut.result()
            if not ok:
                failed.append(sid)
    if failed:
        print(f"{len(failed)} images failed to download. Check and retry.")
    return failed


# -----------------------------
# IMAGE PREPROCESSING + TRANSFORM
# -----------------------------
def get_transform(img_size=224):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def load_image_tensor(path, transform):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img)
    except:
        return None


# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def get_backbone(model_name='resnet50', pretrained=True):
    model_name = model_name.lower()
    if model_name.startswith('resnet'):
        if model_name == 'resnet50':
            m = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet18':
            m = models.resnet18(pretrained=pretrained)
        else:
            raise ValueError("Unsupported ResNet variant")
        feat_dim = m.fc.in_features
        modules = list(m.children())[:-1]  # remove final fc
        backbone = torch.nn.Sequential(*modules)
        return backbone, feat_dim
    else:
        raise ValueError("Unsupported model")

def extract_image_embeddings(image_dir, out_path, model_name='resnet50', batch_size=64, device=None, img_size=224):
    """
    Extract embeddings for all images in image_dir and save as npz + csv
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    backbone, feat_dim = get_backbone(model_name)
    backbone.eval().to(device)
    transform = get_transform(img_size)

    files = sorted(Path(image_dir).glob('*.jpg'))
    ids = [p.stem for p in files]
    n = len(files)
    feats = np.zeros((n, feat_dim), dtype=np.float32)

    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc='Extracting embeddings'):
            batch_files = files[i:i+batch_size]
            tensors = []
            valid_ids = []
            for p in batch_files:
                t = load_image_tensor(str(p), transform)
                if t is not None:
                    tensors.append(t)
                    valid_ids.append(p.stem)
            if not tensors:
                continue
            x = torch.stack(tensors).to(device)
            out = backbone(x)
            out = out.view(out.size(0), -1).cpu().numpy()
            start = i
            for j, sid in enumerate(valid_ids):
                feats[start + j, :] = out[j]

    # Save npz
    base = os.path.splitext(out_path)[0]
    npz_path = base + '.npz'
    csv_path = base + '.csv'
    np.savez_compressed(npz_path, ids=np.array(ids), feats=feats)

    # Save csv
    df = pd.DataFrame(feats)
    df.insert(0, 'sample_id', ids)
    df.to_csv(csv_path, index=False)
    print(f"Saved embeddings: {npz_path} and {csv_path}")


def load_embeddings(npz_or_csv):
    if npz_or_csv.endswith('.npz'):
        d = np.load(npz_or_csv)
        return d['ids'].astype(str), d['feats']
    else:
        df = pd.read_csv(npz_or_csv)
        ids = df['sample_id'].astype(str).values
        feats = df.drop(columns=['sample_id']).values
        return ids, feats


# -----------------------------
# CLI
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['download','extract'], required=True)
    parser.add_argument('--csv', type=str, help='CSV with sample_id + image_link (for download)')
    parser.add_argument('--image_dir', type=str, help='Directory with images (for extract)')
    parser.add_argument('--out', type=str, help='Output npz file path (for extract)')
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    if args.mode == 'download':
        if not args.csv or not args.image_dir:
            raise ValueError("Provide --csv and --image_dir for download")
        df = pd.read_csv(args.csv)
        download_images(df, image_dir=args.image_dir, num_workers=args.num_workers)
    elif args.mode == 'extract':
        if not args.image_dir or not args.out:
            raise ValueError("Provide --image_dir and --out for extract")
        extract_image_embeddings(args.image_dir, args.out, model_name=args.model,
                                 batch_size=args.batch, img_size=args.img_size)
