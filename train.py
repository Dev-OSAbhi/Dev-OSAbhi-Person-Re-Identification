import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from utils.data import DataConfig, UnifiedReIDDataModule, UnifiedReIDDataset
from utils.models import ReIDSystem


TRACK_MODALITIES = {
    "rgb": {"data_modality": "rgb", "anchor": "rgb", "positive": "rgb", "negative": "rgb"},
    "depth": {"data_modality": "depth", "anchor": "depth", "positive": "depth", "negative": "depth"},
    "cross": {"data_modality": "rgbd", "anchor": "rgb", "positive": "depth", "negative": "depth"},
}


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    # Determine data modality based on track
    # Logic in utils/models.py handles "cross" by expecting "rgbd" (both) or similar?
    # No, utils/models.py ReIDSystem checks `self.hparams.modality == "cross"`.
    # utils/data.py UnifiedReIDDataset takes `modality="rgb"|"depth"|"rgbd"`.
    
    # If track is cross, we set data modality to "rgbd" (aka cross load both).
    # ReIDSystem needs to know it is in "cross" mode to use 2 encoders.
    # The config `cfg.track` gives us "rgb", "depth", "cross".
    
    # Map track to data modality
    if cfg.track == "cross":
        data_modality = "rgbd" # Load both
        model_modality = "cross"
    else:
        data_modality = cfg.track
        model_modality = cfg.track
        
    # Instantiate Data Config
    data_cfg: DataConfig = instantiate(cfg.data)
    data_cfg.modality = data_modality
    
    dm = UnifiedReIDDataModule(data_cfg)
    dm.setup("fit")
    
    # Get num_classes from dataset
    # We must access train_set from dm
    # dm.train_set is set after setup("fit")
    if dm.train_set is None:
        raise ValueError("DataModule train_set is None after setup!")
        
    num_classes = len(dm.train_set.pids)
    print(f"Detected {num_classes} identities for training.")
    
    # Steps per epoch for scheduler
    train_loader = dm.train_dataloader()
    steps_per_epoch = len(train_loader)
    
    model = ReIDSystem(
        arch=cfg.model.arch,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
        margin=cfg.model.margin,
        num_classes=num_classes,
        modality=model_modality,
        steps_per_epoch=steps_per_epoch
    )

    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
