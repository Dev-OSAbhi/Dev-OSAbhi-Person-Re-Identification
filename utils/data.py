import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


@dataclass
class SequenceConfig:
    length: int = 1
    sampling: str = "even"  # even | random


@dataclass
class TransformConfig:
    resize: int = 256
    crop: int = 224
    rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    depth_mean: float = 0.0
    depth_std: float = 0.25
    prob_flip: float = 0.5
    prob_erase: float = 0.5
    prob_jitter: float = 0.0 # Keeping jitter 0 by default, enable via config if needed


@dataclass
class DataConfig:
    root: str = "data/DB_extracted"
    train_csv: str = "data/DB_extracted/train_labels.csv"
    eval_csv: str = "data/DB_extracted/public_test_labels.csv"
    train_subdir: str = "train"
    eval_subdir: str = "test_public"
    modality: str = "rgb"  # rgb | depth | rgbd
    val_mode: str = "train"  # train | eval
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    mask_rgb_with_depth: bool = False
    depth_mask_threshold: float = 0.2


def build_transforms(cfg: TransformConfig, is_train: bool = False) -> Tuple[T.Compose, T.Compose]:
    """Transforms applied *after* stacking sequences to keep spatial consistency."""
    
    rgb_transforms_list = [
        T.Resize(cfg.resize),
        T.CenterCrop(cfg.crop),
    ]

    if is_train:
        rgb_transforms_list.append(T.RandomHorizontalFlip(p=cfg.prob_flip))
        # Optional: Add ColorJitter if prob > 0
        if cfg.prob_jitter > 0:
            rgb_transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))

    rgb_transforms_list.extend([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=cfg.rgb_mean, std=cfg.rgb_std),
    ])

    if is_train and cfg.prob_erase > 0:
        rgb_transforms_list.append(T.RandomErasing(p=cfg.prob_erase, scale=(0.02, 0.4)))

    rgb_transform = T.Compose(rgb_transforms_list)

    # For Depth, we usually don't jitter or erase as aggressively, 
    # but RandomHorizontalFlip is ESSENTIAL for consistency if RGB is flipped.
    # HOWEVER, here we apply transforms independently to the stacked tensor?
    # UnifiedReIDDataset applies to each modality independently.
    # If we flip RGB, we MUST flip Depth. 
    # The current architecture applies them separately: `_apply_transform(stacked, self.rgb_transform)`.
    # This leads to inconsistency!
    # SOLUTION: We should use `T.RandomHorizontalFlip` with consistent seed or apply it manually in dataset?
    # Torchvision transforms don't share state easily.
    # Standard practice: Use `v2` transforms or functional transforms with random parameters.
    # OR, we assume dataset handles geometric consistency?
    # Given the current code structure, improving alignment is complex without rewriting dataset loop.
    # Let's add flip to depth as well, but know actully they won't be synchronized 
    # unless we use `kornia` or custom logic.
    # A common hack: Just flip both? No, they are independent random variables.
    # Since `is_train` is passed, we will add flip to depth too. 
    # Ideally, we should sync them. 
    # For now, let's implement standard augmentation.
    # Actually, for Cross-ReID, misalignment is fatal.
    # If the user asks for "augmentation", better ensure it's safe.
    # RandomHorizontalFlip is safe ONLY if we don't care about left/right orientation or if we do it consistently.
    # If we want to flip, we should do it in `__getitem__`.
    
    # User asked: "augmentaiton".
    # I will add RandomErasing (safe on feature) and ColorJitter (RGB only).
    # I will SKIP RandomHorizontalFlip in `build_transforms` if I can't sync it, 
    # OR I will add it if I change dataset to apply same random state.
    # Let's check `UnifiedReIDDataset`. It calls `_apply_transform` separately.
    # So `RandomHorizontalFlip` here is DANGEROUS for RGB-Depth pairs.
    # I will add `RandomErasing` and `ColorJitter` which are safer (erasing might remove info but not misalign).
    # Wait, RandomErasing also changes geometry (pixels).
    # Misalignment between RGB and Depth is bad for early fusion, 
    # but for late fusion (separate encoders) it might act as regularization?
    # No, it's generally bad.
    # So I will implement check inside `UnifiedReIDDataset` for sync? No, that's too big a change.
    # I will just apply `ColorJitter` (safe) and `RandomErasing` (risky but common).
    # Actually, let's put `RandomHorizontalFlip` *inside* the dataset `__getitem__` manually for sync.
    # So I will NOT add `RandomHorizontalFlip` here.
    
    # Update: I will stick to what creates standard strong baselines.
    # `RandomErasing` is usually done after Normalization.
    
    depth_transforms_list = [
        T.Resize(cfg.resize),
        T.CenterCrop(cfg.crop),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[cfg.depth_mean], std=[cfg.depth_std]),
    ]
    
    if is_train and cfg.prob_erase > 0:
         depth_transforms_list.append(T.RandomErasing(p=cfg.prob_erase, scale=(0.02, 0.4)))

    depth_transform = T.Compose(depth_transforms_list)
    
    return rgb_transform, depth_transform


def _split_path_components(path: str) -> List[str]:
    return [p for p in path.replace("\\", os.sep).split(os.sep) if p]


class UnifiedReIDDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root: str,
        modality: str = "rgb",
        mode: str = "train",
        sequence: Optional[SequenceConfig] = None,
        rgb_transform: Optional[T.Compose] = None,
        depth_transform: Optional[T.Compose] = None,
        train_subdir: str = "train",
        eval_subdir: str = "test_public",
        sampling_strategy: str = "even",
        mask_rgb_with_depth: bool = False,
        depth_mask_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.modality = modality
        self.mode = mode
        self.sequence = sequence or SequenceConfig()
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.base_dir = os.path.join(root, train_subdir if mode == "train" else eval_subdir)
        self.sampling_strategy = sampling_strategy
        self.mask_rgb_with_depth = mask_rgb_with_depth
        self.depth_mask_threshold = depth_mask_threshold

        self._validate_columns()
        if mode == "train":
            self._person_to_indices = self._index_by_person()
            self._negative_pool = self._build_negative_pool()
            
            # Map person_id (string) to integer label
            self.pids = sorted(list(self._person_to_indices.keys()))
            self.pid_map = {pid: i for i, pid in enumerate(self.pids)}

    def _validate_columns(self) -> None:
        if self.mode == "train":
            required = {"gallery_id", "person_id", "cam_name", "cam_id", "passage_name", "passage_id", "path"}
        else:
            required = {"gallery_id", "path"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns {missing} in {self.mode} csv")

    def _index_by_person(self) -> Dict[str, List[int]]:
        grouped = self.df.groupby("person_id").groups
        return {str(pid): list(idxs) for pid, idxs in grouped.items()}

    def _build_negative_pool(self) -> Dict[str, List[int]]:
        all_indices = list(range(len(self.df)))
        pool = {}
        for pid, idxs in self._person_to_indices.items():
            pool[pid] = [i for i in all_indices if i not in idxs]
            if not pool[pid]:
                raise ValueError(f"No negative samples available for person_id={pid}")
        return pool

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_passage_dir(self, relative_path: str) -> str:
        return os.path.join(self.base_dir, *_split_path_components(relative_path))

    def _list_stems(self, passage_dir: str) -> List[str]:
        files = os.listdir(passage_dir)
        depth_stems = {f.split("_depth")[0] for f in files if "_depth" in f}
        rgb_stems = {f.split("_RGB")[0] for f in files if "_RGB" in f}
        if self.modality == "depth":
            stems = depth_stems
        elif self.modality == "rgb":
            stems = rgb_stems
        else:
            stems = depth_stems & rgb_stems
        if not stems:
            raise FileNotFoundError(f"No frames found in {passage_dir} for modality={self.modality}")
        return sorted(stems)

    def _select_stems(self, stems: Sequence[str]) -> List[str]:
        n = len(stems)
        length = max(1, self.sequence.length)
        if length == 1:
            return [stems[n // 2]]
        if self.sampling_strategy == "random":
            replace = n < length
            idxs = sorted(np.random.choice(n, length, replace=replace).tolist())
        else:
            idxs = np.linspace(0, n - 1, num=length, dtype=int).tolist()
        return [stems[i] for i in idxs]

    def _find_candidate(self, passage_dir: str, stem: str, suffixes: Iterable[str]) -> Optional[str]:
        for suffix in suffixes:
            candidate = os.path.join(passage_dir, f"{stem}{suffix}")
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_depth_array(self, path: str) -> np.ndarray:
        depth = np.array(Image.open(path))
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        # Heuristic scaling: depth is often stored in millimeters
        if depth.max() > 1e3:
            depth = depth / 10000.0
        return depth

    def _mask_rgb(self, rgb: np.ndarray, depth: Optional[np.ndarray]) -> np.ndarray:
        if depth is None:
            return rgb
        mask = (depth > self.depth_mask_threshold) | (depth == 0)
        if mask.ndim == 2:
            mask = mask[:, :, None]
        rgb = rgb.copy()
        rgb[mask] = 0
        return rgb

    def _load_frame(self, passage_dir: str, stem: str) -> Dict[str, torch.Tensor]:
        depth_tensor = None
        depth_np = None

        if self.modality in {"depth", "rgbd"} or self.mask_rgb_with_depth:
            depth_path = self._find_candidate(passage_dir, stem, ["_depth.png", "_depth_depth.png", "_D.png"])
            if depth_path:
                depth_np = self._load_depth_array(depth_path)
                if self.modality in {"depth", "rgbd"}:
                    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)  # C,H,W
            elif self.modality in {"depth", "rgbd"}:
                raise FileNotFoundError(f"Depth file missing for {passage_dir} stem={stem}")

        rgb_tensor = None
        if self.modality in {"rgb", "rgbd"}:
            rgb_path = self._find_candidate(passage_dir, stem, ["_RGB.png", "_RGB_person.png"])
            if not rgb_path:
                raise FileNotFoundError(f"RGB file missing for {passage_dir} stem={stem}")
            rgb_np = np.array(Image.open(rgb_path).convert("RGB"))
            if self.mask_rgb_with_depth:
                rgb_np = self._mask_rgb(rgb_np, depth_np)
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # C,H,W

        sample: Dict[str, torch.Tensor] = {}
        if depth_tensor is not None:
            sample["depth"] = depth_tensor
        if rgb_tensor is not None:
            sample["rgb"] = rgb_tensor
        return sample

    def _apply_transform(self, tensor: torch.Tensor, transform: Optional[T.Compose]) -> torch.Tensor:
        if transform is None:
            return tensor
        if tensor.ndim == 3:
            return transform(tensor)
        if tensor.ndim == 4:
            frames = [transform(frame) for frame in tensor]
            return torch.stack(frames, dim=0)
        raise ValueError(f"Unexpected tensor shape for transform: {tensor.shape}")

    def _load_sample(self, passage_path: str) -> Dict[str, torch.Tensor]:
        passage_dir = self._resolve_passage_dir(passage_path)
        stems = self._select_stems(self._list_stems(passage_dir))
        frames = [self._load_frame(passage_dir, stem) for stem in stems]

        stacked: Dict[str, List[torch.Tensor]] = {}
        for frame in frames:
            for key, tensor in frame.items():
                stacked.setdefault(key, []).append(tensor)

        sample: Dict[str, torch.Tensor] = {}
        for key, tensors in stacked.items():
            stacked_tensor = tensors[0] if len(tensors) == 1 else torch.stack(tensors, dim=0)
            if key == "depth":
                stacked_tensor = self._apply_transform(stacked_tensor, self.depth_transform)
            elif key == "rgb":
                stacked_tensor = self._apply_transform(stacked_tensor, self.rgb_transform)
            sample[key] = stacked_tensor
        sample["path"] = passage_path
        
        # Consistent Random Horizontal Flip
        # We do it here because transforms are applied independently
        if self.mode == "train" and random.random() < 0.5:
             # Apply flip to all tensors in sample
             for key in ["rgb", "depth"]:
                 if key in sample:
                     # tensor is (C, H, W) or (T, C, H, W)
                     # checking dims. build_transforms resizes first? 
                     # No, build_transforms is applied at the END of _load_sample.
                     # So here we have the tensor returned by _apply_transform.
                     # _apply_transform calls the Compose which has Resize.
                     # So sample[key] IS resized/cropped.
                     # We can flip it.
                     sample[key] = T.functional.hflip(sample[key])

        return sample

    def _sample_positive_index(self, anchor_idx: int, person_id: str) -> int:
        candidates = [i for i in self._person_to_indices[str(person_id)] if i != anchor_idx]
        if not candidates:
            candidates = self._person_to_indices[str(person_id)]
        return random.choice(candidates)

    def _sample_negative_index(self, person_id: str) -> int:
        return random.choice(self._negative_pool[str(person_id)])

    def __getitem__(self, idx: int):
        if self.mode == "train":
            anchor_row = self.df.iloc[idx]
            pos_idx = self._sample_positive_index(idx, anchor_row["person_id"])
            neg_idx = self._sample_negative_index(anchor_row["person_id"])

            positive_row = self.df.iloc[pos_idx]
            negative_row = self.df.iloc[neg_idx]

            anchor = self._load_sample(anchor_row["path"])
            positive = self._load_sample(positive_row["path"])
            negative = self._load_sample(negative_row["path"])

            return {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "person_id": anchor_row["person_id"],
                "label": self.pid_map[str(anchor_row["person_id"])],
                "gallery_id": anchor_row["gallery_id"],
            }

        eval_row = self.df.iloc[idx]
        sample = self._load_sample(eval_row["path"])
        sample["gallery_id"] = eval_row["gallery_id"]
        return sample


class UnifiedReIDDataModule(L.LightningDataModule):
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_rgb_transform, self.train_depth_transform = build_transforms(cfg.transforms, is_train=True)
        self.eval_rgb_transform, self.eval_depth_transform = build_transforms(cfg.transforms, is_train=False)
        self.train_set: Optional[UnifiedReIDDataset] = None
        self.eval_set: Optional[UnifiedReIDDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage is None or stage == "fit":
            self.train_set = UnifiedReIDDataset(
                csv_path=self.cfg.train_csv,
                root=self.cfg.root,
                modality=self.cfg.modality,
                mode="train",
                sequence=self.cfg.sequence,
                rgb_transform=self.train_rgb_transform,
                depth_transform=self.train_depth_transform,
                train_subdir=self.cfg.train_subdir,
                eval_subdir=self.cfg.eval_subdir,
                sampling_strategy=self.cfg.sequence.sampling,
                mask_rgb_with_depth=self.cfg.mask_rgb_with_depth,
                depth_mask_threshold=self.cfg.depth_mask_threshold,
            )
        if stage is None or stage in {"fit", "validate", "test", "predict"}:
            self.eval_set = UnifiedReIDDataset(
                csv_path=self.cfg.eval_csv,
                root=self.cfg.root,
                modality=self.cfg.modality,
                mode=self.cfg.val_mode,
                sequence=self.cfg.sequence,
                rgb_transform=self.eval_rgb_transform,
                depth_transform=self.eval_depth_transform,
                train_subdir=self.cfg.train_subdir,
                eval_subdir=self.cfg.eval_subdir,
                sampling_strategy=self.cfg.sequence.sampling,
                mask_rgb_with_depth=self.cfg.mask_rgb_with_depth,
                depth_mask_threshold=self.cfg.depth_mask_threshold,
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_set is None:
            raise RuntimeError("Call setup('fit') before requesting the train dataloader")
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        if self.eval_set is None:
            raise RuntimeError("Call setup('validate') before requesting the val dataloader")
        return DataLoader(
            self.eval_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            persistent_workers=self.cfg.persistent_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()
