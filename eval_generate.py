import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import lightning as L
import torch.nn.functional as F

from utils.models import ReIDSystem
from utils.data import UnifiedReIDDataset, build_transforms, TransformConfig, DataConfig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs='+', required=True, help="List of checkpoint paths")
    parser.add_argument("--track", type=str, required=True, choices=["rgb", "depth", "cross"])
    parser.add_argument("--data-root", type=str, default="data/DB_extracted")
    parser.add_argument("--labels-csv", type=str, default="data/DB_extracted/public_test_labels.csv")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def extract_features(model, loader, device, modality):
    model.eval()
    features = []
    gallery_ids = []
    paths = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting"):
            # batch is from UnifiedReIDDataset evaluation mode
            # keys: rgb, depth, path, gallery_id (and potentially others)
            
            g_ids = batch['gallery_id']
            b_paths = batch['path']
            
            # Determine input tensor based on modality and model expectation
            # ReIDSystem.forward(x, input_modality)
            
            # If track is cross, we ideally use the "best" available modality or both?
            # In evaluation (UnifiedReIDDataset): 
            # If modality="cross" (or "rgbd"), we have both.
            # If standard cross-reid eval:
            # We usually extract features for whatever is available.
            # But here `loader` gives us a batch.
            # For "cross" track, the competition likely evaluates RGB queries vs Depth gallery (or vice versa).
            # But the provided CSV is a list of images.
            # If we want to support generic "Cross" features, we should probably output 
            # a concatenation of RGB-feature and Depth-feature if both exist?
            # Or if the model is "ReIDSystem" with "cross" modality, it has 2 encoders.
            # We should probably pass both if available.
            
            # Our ReIDSystem forward: forward(x, input_modality="rgb")
            # If model is cross, it dispatches to specific encoder.
            
            # Strategy for Ensemble:
            # We extract feature vector for each image.
            # If model is cross-modal, we might get 2 vectors (one from RGB enc, one from Depth enc)?
            # Or we just use the one corresponding to the image type?
            # The dataset `modality` determines what is loaded.
            # If track="cross", dataset loads both.
            
            # Let's assume for Cross Track:
            # We compute feature = Normalize(Enc_RGB(rgb)) + Normalize(Enc_Depth(depth)) ?
            # Or Concat?
            # Existing literature (e.g. standard baseline):
            # If query is RGB, use Enc_RGB. If gallery is Depth, use Enc_Depth.
            # But we are generating a single CSV with "distance".
            # We compute distance matrix between ALL images in the list.
            # So we need a compatible feature space.
            # If we map RGB->Shared and Depth->Shared, then we can compute distance.
            # So we just need to forward through the correct encoder.
             
            # How do we know if an image in batch is "RGB" or "Depth"?
            # The dataset doesn't explicitly flag "this is an RGB image" in the batch 
            # unless we inferred it from path or it's fixed.
            # BUT UnifiedReIDDataset loads PAIRS for "cross" modality (same image, both views).
            # Wait, `UnifiedReIDDataset` for eval just loads the sample at `path`.
            # If `path` points to a passage, it loads frames.
            # If `modality='cross'`, it loads RGB AND Depth frames for that passage.
            # So we have BOTH.
            # We should probably AVG the features from both encoders to get maximum robust representation?
            # Yes, fusion (AVG or CONCAT) is best.
            
            curr_feats = []
            
            # 1. RGB Feature
            if 'rgb' in batch:
                x = batch['rgb'].to(device)
                f = model(x, input_modality="rgb")
                f = F.normalize(f, p=2, dim=1)
                curr_feats.append(f)
            
            # 2. Depth Feature
            if 'depth' in batch:
                x = batch['depth'].to(device)
                f = model(x, input_modality="depth")
                f = F.normalize(f, p=2, dim=1)
                curr_feats.append(f)
                
            if not curr_feats:
                # Should not happen
                continue
                
            # Combine
            if len(curr_feats) > 1:
                # Average features (late fusion)
                final_f = torch.stack(curr_feats).mean(dim=0)
            else:
                final_f = curr_feats[0]
                
            final_f = F.normalize(final_f, p=2, dim=1)
            features.append(final_f.cpu())
            gallery_ids.extend(g_ids)
            paths.extend(b_paths)
            
    features = torch.cat(features, dim=0)
    return features, gallery_ids, paths

def main():
    args = get_args()
    device = args.device
    print(f"Using device: {device}")
    
    # 1. Setup Dataset
    # Decide modality for dataset
    if args.track == "cross":
        modality = "rgbd" # Triggers loading both if available
    else:
        modality = args.track
        
    cfg = TransformConfig() 
    # Valid transforms (Resize only, no flip/erase)
    rgb_t, depth_t = build_transforms(cfg, is_train=False)
    
    ds = UnifiedReIDDataset(
        csv_path=args.labels_csv,
        root=args.data_root,
        mode="eval",
        modality=modality,
        rgb_transform=rgb_t,
        depth_transform=depth_t
    )
    
    loader = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    ensemble_features = []
    
    # 2. Loop Checkpoints
    for ckpt in args.checkpoints:
        print(f"Processing checkpoint: {ckpt}")
        try:
            model = ReIDSystem.load_from_checkpoint(ckpt)
        except Exception as e:
            print(f"Failed to load {ckpt}: {e}")
            continue
            
        model.to(device)
        model.eval()
        
        feats, g_ids, paths = extract_features(model, loader, device, modality)
        ensemble_features.append(feats)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
        
    if not ensemble_features:
        print("No features extracted!")
        return

    # 3. Concatenate Features (Ensemble)
    print("Combining features from ensemble...")
    # Normalize again just in case
    ensemble_features = [F.normalize(f, p=2, dim=1) for f in ensemble_features]
    # Concatenate columns: [N, D1+D2+...]
    # This weights all models equally (if dims are similar) or by dimension.
    # To weight equally regardless of dim, we normalized.
    # Concat is standard.
    final_features = torch.cat(ensemble_features, dim=1) # (N, TotalDim)
    
    # 4. Compute Distance
    print("Computing distance matrix...")
    # (N, TotalDim) * (TotalDim, N) -> (N, N)
    # Cosine similarity
    sim = torch.mm(final_features, final_features.t())
    
    # Convert to distance. 
    # Note: If we concatenated K normalized vectors, the dot product range is [-K, K].
    # We want distance.
    # Standard: dist = K - sim ? Or just sort by sim (descending).
    # Task requires "distance" column. 
    # Usually 1 - cosine.
    # Here max sim is K (if 1 model, 1).
    # Let's normalize final_features again?
    # No, that changes the metric.
    # Distance = 1 - (sim / K).
    K = len(ensemble_features)
    dist = 1.0 - (sim / K)
    
    # 5. Generate CSV
    print(f"Generating CSV to {args.output}...")
    
    with open(args.output, "w") as f:
        f.write("query_gallery_id,query_path,gallery_id,gallery_path,rank,distance\n")
        
        N = len(final_features)
        for i in range(N):
            q_gid = g_ids[i]
            # q_path = paths[i] # Full path
            
            dists = dist[i]
            indices = torch.argsort(dists) # Ascending distance
            
            rank = 1
            for idx in indices:
                idx = idx.item()
                if idx == i:
                    continue
                
                # Format paths to relative if needed (hacky consistent with prev)
                q_p = paths[i].split("data/DB_extracted/")[-1] 
                g_p = paths[idx].split("data/DB_extracted/")[-1] 
                
                d_val = dists[idx].item()
                
                f.write(f"{q_gid},{q_p},{g_ids[idx]},{g_p},{rank},{d_val:.6f}\n")
                rank += 1
                
    print("Done.")

if __name__ == "__main__":
    main()
