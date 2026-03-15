import argparse
import os
import os.path as osp
import glob
import csv
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from torchreid import models
from torchreid import models
from torchreid.utils import load_pretrained_weights
from torchreid.utils import load_pretrained_weights
import cv2

def get_images(passage_dir, modality):
    # Modality: 'rgb' or 'depth'
    valid_files = []
    if not osp.exists(passage_dir):
        return []
    
    files = os.listdir(passage_dir)
    for f in files:
        if modality == 'rgb' and '_RGB.png' in f:
            valid_files.append(osp.join(passage_dir, f))
        elif modality == 'depth' and ('_depth.png' in f or '_D.png' in f):
            valid_files.append(osp.join(passage_dir, f))
    return sorted(valid_files)

def load_and_preprocess(image_paths, transform, device, modality='rgb'):
    imgs = []
    for p in image_paths:
        try:
            if modality == 'depth':
                # Apply JET Colormap for depth
                if not osp.exists(p):
                     print(f"Error loading {p}: does not exist")
                     continue
                img_gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img_gray is None:
                    print(f"Error loading {p}: read failed")
                    continue
                img_color = cv2.applyColorMap(img_gray, cv2.COLORMAP_JET)
                img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img_color)
            else:
                # RGB
                img = Image.open(p).convert('RGB')
            
            img = transform(img)
            imgs.append(img)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    if not imgs:
        return None
    imgs = torch.stack(imgs).to(device)
    return imgs

def extract_features(model, images):
    # content: [B, C, H, W]
    # return: [D] (mean feature)
    with torch.no_grad():
        features = model(images) # [B, D]
        # Normalize? standard torchreid models might not normalize in forward if it's just 'v'.
        # But we usually want Euclidean distance on unnormalized or normalized?
        # ReID usually uses cosine distance or Euclidean on normalized features.
        # Let's normalize.
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        mean_feat = features.mean(dim=0)
        mean_feat = torch.nn.functional.normalize(mean_feat, p=2, dim=0)
        return mean_feat

def main():
    parser = argparse.ArgumentParser(description='Generate TVRID submission')
    parser.add_argument('--root', type=str, default='data/DB_extracted', help='dataset root')
    parser.add_argument('--csv', type=str, default='data/DB_extracted/public_test_labels.csv', help='test csv')
    parser.add_argument('--track', type=str, default='rgb', choices=['rgb', 'depth', 'cross'])
    parser.add_argument('--checkpoint', type=str, required=True, help='path to trained model weights')
    parser.add_argument('--model', type=str, default='osnet_x1_0', help='model name')
    parser.add_argument('--output', type=str, default='submission.csv', help='output csv path')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch-size', type=int, default=1) # passages vary in length, batch=1 passage easier
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data setup
    df = pd.read_csv(args.csv)
    test_dir = osp.join(args.root, 'test_public')
    
    # Transform
    # Standard ReID transform for testing
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Model
    # We don't know num_classes from test CSV. 
    # But for inference it doesn't matter (classifier layer ignored).
    # Pass a dummy num_classes.
    model = models.build_model(
        name=args.model,
        num_classes=100, 
        loss='softmax',
        pretrained=False
    )
    load_pretrained_weights(model, args.checkpoint)
    model.to(device)
    model.eval()

    # Extract features
    query_feats = []
    gallery_feats = []
    ids = []
    paths = []

    print(f"Extracting features for track: {args.track}")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        gallery_id = str(row['gallery_id'])
        rel_path = row['path']
        ids.append(gallery_id)
        paths.append(rel_path)
        
        passage_dir = osp.join(test_dir, rel_path)
        
        # Determine what to load
        # track=rgb: Q=RGB, G=RGB
        # track=depth: Q=Depth, G=Depth
        # track=cross: Q=RGB, G=Depth
        
        load_rgb = args.track in ['rgb', 'cross']
        load_depth = args.track in ['depth', 'cross']
        
        q_feat = None
        g_feat = None
        
        if load_rgb:
            imgs = load_and_preprocess(get_images(passage_dir, 'rgb'), transform, device)
            if imgs is not None:
                feat = extract_features(model, imgs)
                if args.track == 'rgb':
                    q_feat = feat
                    g_feat = feat
                elif args.track == 'cross':
                    q_feat = feat
                    
        if load_depth:
            imgs = load_and_preprocess(get_images(passage_dir, 'depth'), transform, device, modality='depth')
            if imgs is not None:
                feat = extract_features(model, imgs)
                if args.track == 'depth':
                    q_feat = feat
                    g_feat = feat
                elif args.track == 'cross':
                    g_feat = feat
        
        # Handle missing data (should not happen in valid dataset)
        if q_feat is None: q_feat = torch.zeros(512).to(device) # dummy
        if g_feat is None: g_feat = torch.zeros(512).to(device) # dummy
        
        query_feats.append(q_feat)
        gallery_feats.append(g_feat)

    query_tensor = torch.stack(query_feats) # [N, D]
    gallery_tensor = torch.stack(gallery_feats) # [N, D]
    
    print("Computing distance matrix...")
    # Euclidean distance
    # dist[i, j] = ||q_i - g_j||
    dists = torch.cdist(query_tensor, gallery_tensor).cpu()
    
    print("Generating rankings...")
    rankings = []
    
    N = len(ids)
    for i in tqdm(range(N)):
        d_row = dists[i]
        # Sort
        sorted_indices = torch.argsort(d_row)
        
        rank = 1
        query_id = ids[i]
        query_path = paths[i]
        
        for j in sorted_indices:
            gallery_idx = j.item()
            target_id = ids[gallery_idx]
            
            if target_id == query_id:
                continue # skip self
                
            rankings.append({
                "query_gallery_id": query_id,
                "query_path": query_path,
                "gallery_id": target_id,
                "gallery_path": paths[gallery_idx],
                "rank": rank,
                "distance": float(d_row[gallery_idx].item())
            })
            rank += 1
            
    # Write CSV
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["query_gallery_id", "query_path", "gallery_id", "gallery_path", "rank", "distance"],
        )
        writer.writeheader()
        writer.writerows(rankings)
    print(f"Done. Saved to {args.output}")

if __name__ == '__main__':
    main()
