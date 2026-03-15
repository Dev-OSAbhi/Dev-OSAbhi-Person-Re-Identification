# ICPR-TVRID
Competition on Privacy-Preserving Person Re-Identification from Top-View RGB-Depth Camera (TVRID).

## Overview
TVRID is the ICPR 2026 competition on top-view person re-identification with aligned RGB and Depth. The benchmark captures 88 identities with four overhead Intel RealSense D455 cameras observing each passage twice (IN/OUT) across four geometric contexts: flat ground, ascent, descent, and oblique roof view. Submissions are ranked lists evaluated with CMC@1/5/10 and mAP, and the primary leaderboard metric is `overallMap` (mean of per-track overall mAP).

Tracks:
- RGB Re-ID (privacy-unconstrained)
- Depth Re-ID (privacy-preserving)
- Cross-modal RGB↔Depth retrieval

Competition page: https://www.codabench.org/competitions/12315/

## Data download and layout
- Download from Zenodo: https://zenodo.org/records/17909410 (DOI: 10.5281/zenodo.17909410). Two zips are provided:
  - **Original**: full-resolution RGB/Depth streams.
  - **Extracted**: depth-guided crops (300×300) centered on each pedestrian—recommended for quick start and matches this repo’s defaults.
- Unzip the **extracted** archive into `data/DB_extracted` so the structure matches:
  - `data/DB_extracted/train_labels.csv`
  - `data/DB_extracted/public_test_labels.csv`
  - `data/DB_extracted/train/` (cropped train data)
  - `data/DB_extracted/test_public/` (cropped public test data)

Important note:
- `public_test_labels.csv` is intended for **ranking generation on the public test split**. In a competition setting, test ground truth is not provided locally, so this file must **not** be used as a labeled validation split during training.

If you prefer to re-extract from the original zip, place it separately and adapt paths/transforms as needed. Update paths via CLI flags or Hydra configs if you store data elsewhere.

## Environment
Create and activate a virtual environment, then install requirements:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Baseline training
- Defaults live in `config/train.yaml` and `config/data/data.yaml`.
- `data.eval_csv` is a **validation split used during training**.

Because this is a competition, participants should create their own train/validation split **from `train_labels.csv`**, typically by splitting **by identity** (to avoid identity leakage across splits). The training set contains 62 unique identities; for example, a 50/50 split would allocate 31 identities to train and 31 identities to validation, keeping each identity entirely in one split.

Example (RGB track) with a user-defined split:
```bash
python train.py track=rgb data.root=data/DB_extracted   data.train_csv=data/DB_extracted/train_split.csv   data.eval_csv=data/DB_extracted/valid_split.csv   trainer.max_epochs=20
```

Recommended before generating submission rankings: train on the full training set (maximize data usage). You can set the same CSV for both arguments:
```bash
python train.py track=rgb data.root=data/DB_extracted   data.train_csv=data/DB_extracted/train_labels.csv   data.eval_csv=data/DB_extracted/train_labels.csv   trainer.max_epochs=20
```

Change `track=depth` or `track=cross` to train the other baselines; adjust batch size/devices as needed.

## Generate rankings (submission CSVs)
Use a trained Lightning checkpoint.

You can either:
1) Train the baseline yourself (see section above), or
2) Download the official pretrained baseline checkpoints from the GitHub **Releases** (they are not stored in the repository due to file size constraints).

After downloading, place them in `baselines_weights/` (or update the `--checkpoint` path accordingly).

Expected checkpoint paths:
- `baselines_weights/rgb_weights.ckpt`
- `baselines_weights/depth_weights.ckpt`
- `baselines_weights/cross_weights.ckpt`

Release page (checkpoints):
```text
https://github.com/RaphaelDel/ICPR-2026-TVRID/releases/tag/weights
```

Generate ranking CSVs for Codabench submission (public test split):
```bash
# RGB
python eval_generate.py --checkpoint baselines_weights/rgb_weights.ckpt --track rgb --data-root data/DB_extracted --labels-csv data/DB_extracted/public_test_labels.csv --output outputs/rankings_rgb.csv

# Depth
python eval_generate.py --checkpoint baselines_weights/depth_weights.ckpt --track depth --data-root data/DB_extracted --labels-csv data/DB_extracted/public_test_labels.csv --output outputs/rankings_depth.csv

# Cross-modal (RGB→Depth)
python eval_generate.py --checkpoint baselines_weights/cross_weights.ckpt --track cross --data-root data/DB_extracted --labels-csv data/DB_extracted/public_test_labels.csv --output outputs/rankings_cross.csv
```

Each CSV must contain columns:
`query_gallery_id, query_path, gallery_id, gallery_path, rank, distance`.

## Local evaluation
For sanity checks on your own labeled splits (e.g., derived from `train_labels.csv`), use this two-step workflow.
The example below is for the RGB track, but you can adapt it for Depth and Cross by switching `--track` and output names.

1) Generate rankings on your validation split (stored under `train/` by default):
```bash
python eval_generate.py --checkpoint baselines_weights/rgb_weights.ckpt --track rgb \
  --data-root data/DB_extracted --labels-csv data/DB_extracted/valid_split.csv \
  --eval-subdir train --output outputs/rankings_rgb_valid.csv
```

2) Score those rankings against the same labeled CSV:
```bash
python eval_score.py --rankings outputs/rankings_rgb_valid.csv --secret-map data/DB_extracted/valid_split.csv
```

The official Codabench evaluation uses hidden test labels; your submissions are scored server-side with the same `bundle/scoring_program/scoring.py`, producing `scores.json`, `detailed_results.html`, and the primary `overallMap`.

## Submission packaging
- Create a zip with the three CSV files at the archive root (exact names required):
  - `rankings_rgb.csv`
  - `rankings_depth.csv`
  - `rankings_cross.csv`

Example:
```bash
cd outputs
zip submission.zip rankings_rgb.csv rankings_depth.csv rankings_cross.csv
```

Submit the zip on the Codabench page above. The leaderboard sorts by `overallMap`, computed as the mean of per-track `*mAP`, with CMC@1/5/10 and mAP reported per track and scenario.

## How to participate
- Register on the Codabench competition page and clone this repository.
- Prepare data as described, generate ranked lists for one or more tracks, and package them into the zip format above.
- Upload the zip; Codabench runs the scoring program to produce `scores.json` and leaderboard entries.
- For test phases, only the public labels are available locally; final scoring uses hidden test labels on the server.

## Dataloader quickstart for your own models
The dataset interface lives in `utils/data.py` and is meant to be reusable beyond the provided Lightning baselines.

- Core pieces: `DataConfig` (paths, modality, sequence length/sampling, normalization), `build_transforms` (per-modality torchvision transforms), and `UnifiedReIDDataset`.
- Modalities: set `modality` to `rgb`, `depth`, or `rgbd` (cross-modal) to control which inputs are loaded. Depth masking of RGB can be enabled with `mask_rgb_with_depth`.
- Sequences: `sequence.length` picks how many frames per passage to sample (evenly or randomly via `sequence.sampling`); transforms are applied after stacking to keep spatial alignment.

CSV requirements:
- For **training** (train/validation splits): CSVs must contain
  `gallery_id, person_id, cam_name, cam_id, passage_name, passage_id, path` (same schema as `train_labels.csv`).
- For **ranking generation on test splits**: the CSV only needs
  `gallery_id, path`.

Paths are relative to `train_subdir` or `eval_subdir`.

Sample usage with plain PyTorch:
```python
from utils.data import DataConfig, UnifiedReIDDataset, build_transforms
from torch.utils.data import DataLoader

cfg = DataConfig(
    root="data/DB_extracted",
    train_csv="data/DB_extracted/train_labels.csv",
    # For training-time validation, provide a labeled split derived from train_labels.csv:
    # eval_csv="data/DB_extracted/valid_split.csv",
    modality="rgb",          # or depth / rgbd
    sequence={"length": 5},  # set >1 for temporal sampling
)
rgb_t, depth_t = build_transforms(cfg.transforms)
train_set = UnifiedReIDDataset(
    csv_path=cfg.train_csv, root=cfg.root, modality=cfg.modality, mode="train",
    sequence=cfg.sequence, rgb_transform=rgb_t, depth_transform=depth_t,
    train_subdir=cfg.train_subdir, eval_subdir=cfg.eval_subdir,
)
loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
batch = next(iter(loader))
rgb = batch.get("rgb")      # [B,C,H,W] if modality includes RGB
depth = batch.get("depth")  # [B,1,H,W] if modality includes depth
```

For quick integration with Lightning, `UnifiedReIDDataModule` (also in `utils/data.py`) wraps the same dataset and applies the config defaults found in `config/data/data.yaml`.
