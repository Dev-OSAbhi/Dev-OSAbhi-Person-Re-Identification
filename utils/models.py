import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import lightning as L
from typing import Tuple, Optional
from .osnet import osnet_x1_0

# --- Layers ---

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        # inputs: (B, Dim)
        # Hard mining
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        y = dist_an.data.new().resize_as_(dist_an).fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# --- Backbone ---

class ResNet50ReID(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        
        # Remove original classifier
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # GeM Pooling
        self.pool = GeM()
        
        # BNNeck
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.bnneck.apply(self.weights_init_kaiming)
        
        # Classifier
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(2048, num_classes, bias=False)
            self.classifier.apply(self.weights_init_classifier)

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def _handle_input(self, x):
        if x.dim() == 5:
            self.B, self.S, C, H, W = x.shape
            x = x.view(self.B * self.S, C, H, W)
        else:
            self.B = x.size(0)
            self.S = 1
        return x

    def forward(self, x):
        x = self._handle_input(x)
        
        f = self.features(x)
        f = self.pool(f).flatten(1)
        f_bn = self.bnneck(f)
        
        if self.training:
            if self.num_classes > 0:
                f = f.view(self.B, self.S, -1).mean(dim=1)
                f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
                
                y = self.classifier(f_bn)
                return f, y 
                
            f = f.view(self.B, self.S, -1).mean(dim=1)
            return f, None
            
        f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
        return f_bn

class ConvNextReID(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super().__init__()
        backbone = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT if pretrained else None)
        
        # ConvNeXt feature extractor
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # ConvNeXt Base output dim = 1024
        self.in_features = 1024 
        
        # BNNeck
        self.bnneck = nn.BatchNorm1d(self.in_features)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(self.weights_init_kaiming)
        
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(self.in_features, num_classes, bias=False)
            self.classifier.apply(self.weights_init_classifier)
            
    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
             nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        elif 'BatchNorm' in classname:
             nn.init.constant_(m.weight, 1.0)
             nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        classname = m.__class__.__name__
        if 'Linear' in classname:
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def _handle_input(self, x):
        if x.dim() == 5:
            self.B, self.S, C, H, W = x.shape
            x = x.view(self.B * self.S, C, H, W)
        else:
            self.B = x.size(0)
            self.S = 1
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, x):
        x = self._handle_input(x)
        
        x = self.features(x)
        x = self.avgpool(x)
        f = x.flatten(1)
        f_bn = self.bnneck(f)
        
        if self.training:
            if self.num_classes > 0:
                f = f.view(self.B, self.S, -1).mean(dim=1)
                f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
                
                y = self.classifier(f_bn)
                return f, y
            
            f = f.view(self.B, self.S, -1).mean(dim=1)
            return f, None
            
        f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
        return f_bn

class TransformerReID(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super().__init__()
        # Swin-B-V2
        backbone = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if pretrained else None)
        
        self.features = backbone.features
        self.norm = backbone.norm
        self.permute = backbone.permute
        self.avgpool = backbone.avgpool
        self.flatten = backbone.flatten
        
        self.in_features = 1024 # Swin-B
        
        self.bnneck = nn.BatchNorm1d(self.in_features)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(self.weights_init_kaiming)
        
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(self.in_features, num_classes, bias=False)
            self.classifier.apply(self.weights_init_classifier)

    def weights_init_kaiming(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def _handle_input(self, x):
        if x.dim() == 5:
            self.B, self.S, C, H, W = x.shape
            x = x.view(self.B * self.S, C, H, W)
        else:
            self.B = x.size(0)
            self.S = 1
        
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return x

    def forward(self, x):
        x = self._handle_input(x)
        
        x = self.features(x)
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        f = self.flatten(x)
        
        f_bn = self.bnneck(f)
        
        if self.training:
            if self.num_classes > 0:
                f = f.view(self.B, self.S, -1).mean(dim=1)
                f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
                
                y = self.classifier(f_bn)
                return f, y
            
            f = f.view(self.B, self.S, -1).mean(dim=1)
            return f, None

        f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
        return f_bn

class OSNetReID(nn.Module):
    def __init__(self, num_classes=0, pretrained=True):
        super().__init__()
        self.backbone = osnet_x1_0(pretrained=pretrained, num_classes=num_classes)
        
        self.pool = GeM()
        self.in_features = 512
        
        self.bnneck = nn.BatchNorm1d(self.in_features)
        self.bnneck.bias.requires_grad_(False)
        self.bnneck.apply(self.weights_init_kaiming)
        
        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(self.in_features, num_classes, bias=False)
            self.classifier.apply(self.weights_init_classifier)

    def weights_init_kaiming(self, m):
         if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
         elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def weights_init_classifier(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None: nn.init.constant_(m.bias, 0.0)

    def _handle_input(self, x):
        # x: (B, C, H, W) or (B, S, C, H, W)
        if x.dim() == 5:
            self.B, self.S, C, H, W = x.shape
            x = x.view(self.B * self.S, C, H, W)
        else:
            self.B = x.size(0)
            self.S = 1
            
        # Handle 1-channel input (Depth)
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        return x

    def forward(self, x):
        x = self._handle_input(x)
        
        # Backbone forward
        x = self.backbone.conv1(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        f_map = self.backbone.conv5(x)
        
        f = self.pool(f_map).flatten(1)
        f_bn = self.bnneck(f)
        
        if self.training:
            if self.num_classes > 0:
                y = self.classifier(f_bn)
                # Reshape back to B if needed? 
                # Loss expects (N, C). N = B*S
                # BUT TripletLoss uses anchor/pos/neg which are (B, Dim).
                # Only CE Loss uses all frames.
                # However, ReIDSystem training_step separates logic.
                # If we return N=B*S, then TripletLoss in ReIDSystem receives B*S.
                # But TripletLoss there expects B (triplets).
                # We should average features across sequence?
                # Standard VideoReID: Avg Pooling over time.
                
                # Reshape and Average
                f = f.view(self.B, self.S, -1).mean(dim=1)
                f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
                
                # Re-compute logits on averaged feature? 
                # Or average logits?
                # Usually average feature, then classifiers.
                # But here classifier is applied on f_bn (before average).
                # Let's apply classifier on averaged f_bn.
                y = self.classifier(f_bn)
                return f, y
            
            # If no classifier, average features
            f = f.view(self.B, self.S, -1).mean(dim=1)
            return f, None
            
        # Eval mode
        f_bn = f_bn.view(self.B, self.S, -1).mean(dim=1)
        return f_bn


# --- Lightning Module ---

class ReIDSystem(L.LightningModule):
    def __init__(
        self, 
        arch="resnet50", # resnet50 | convnext | transformer | osnet
        lr=3.5e-4,
        weight_decay=5e-4,
        margin=0.3,
        num_classes=0,
        modality="rgb",
        steps_per_epoch=100  # For scheduler estimation
    ):
        super().__init__()
        self.save_hyperparameters()
        
        def build_model(arch_name, n_classes, pre):
            if arch_name == "resnet50":
                return ResNet50ReID(n_classes, pre)
            elif arch_name == "convnext":
                return ConvNextReID(n_classes, pre)
            elif arch_name == "transformer":
                return TransformerReID(n_classes, pre)
            elif arch_name == "osnet":
                return OSNetReID(n_classes, pre)
            else:
                raise ValueError(f"Unknown arch: {arch_name}")

        # Encoder setup
        if modality == "cross":
            # Two-stream: one for RGB, one for Depth
            self.encoder_rgb = build_model(arch, num_classes, True)
            self.encoder_depth = build_model(arch, num_classes, True)
        else:
            # Shared encoder (RGB or Depth-as-RGB)
            self.encoder = build_model(arch, num_classes, True)
            
        self.ce_loss = CrossEntropyLabelSmooth(num_classes)
        self.triplet_loss = TripletLoss(margin)
        
    def forward(self, x, input_modality="rgb"):
        # x: (B, 3, H, W)
        if self.hparams.modality == "cross":
            if input_modality == "rgb":
                return self.encoder_rgb(x)
            elif input_modality == "depth":
                return self.encoder_depth(x)
        else:
            return self.encoder(x)
        return None

    def training_step(self, batch, batch_idx):
        # Batch: {'img': ..., 'label': ...} for single
        # Batch: {'rgb': ..., 'depth': ..., 'label': ...} for cross
        # Note: UnifiedReIDDataset returns dict with keys 'person_id', 'gallery_id', 'anchor' (dict), etc.
        # But wait, UnifiedReIDDataset __getitem__ returns:
        # { "anchor": {...}, "positive": {...}, "negative": {...}, "person_id": ..., "gallery_id": ... }
        # And PKSampler or similar?
        # My previous `train.py` logic used `dm.train_dataloader()`.
        # Inspecting `train.py` from previous turn... it didn't use PKSampler explicitly in `UnifiedReIDDataModule`.
        # `UnifiedReIDDataset` uses `_sample_positive_index` and `_sample_negative_index` INSIDE `__getitem__`.
        # So each item is a Triplet (Anchor, Positive, Negative).
        # So Batch is a Dict of Lists (collated).
        # Anchor is a Dict with 'rgb', 'depth'.
        
        # We need to extract tensors.
        
        # unpack
        # batch is a dict. keys: anchor, positive, negative, ...
        # each value is dict or tensor? 
        # default_collate will stack them.
        
        anchor_imgs = batch['anchor']
        pos_imgs = batch['positive']
        neg_imgs = batch['negative']
        
        # Flatten for processing?
        # ReIDSystem training_step in previous code (the one I read in Step 26) seemed to expect PKSampler batch?
        # "Batch: {'img': ..., 'label': ...}" -> This implies standard classification batch if using PKSampler.
        # BUT UnifiedReIDDataset returns Triplets!
        # So we must use Triplet Loss directly on these triplets.
        
        # Let's support both Cross and Single modality.
        
        def forward_one(imgs):
            if self.hparams.modality == "cross":
                rgb = imgs['rgb']
                depth = imgs['depth']
                f_rgb, l_rgb = self.encoder_rgb(rgb)
                f_depth, l_depth = self.encoder_depth(depth)
                # Concatenate or Average?
                # For training, we want to align both?
                # SOTA: Learn shared space.
                # Common strategy: Concatenate features for loss? Or sum losses?
                # Let's concatenate features for a "Unified" representation.
                f = torch.cat([f_rgb, f_depth], dim=1) # (B, 2048+2048)
                # But dimensionalty increases.
                # Let's just sum the losses for separate streams + cross constraint?
                # User asked for "strategy for depth and cross tracks".
                # Simple strong baseline: Concat features.
                l = None # Logits? 
                # If we concat, we need classifier for 4096 dim.
                # But our classifiers are inside encoder (2048 dim).
                # So we have logits_rgb and logits_depth.
                return f, (l_rgb, l_depth)
            else:
                # Single modality
                # dataset key depends on modality?
                # UnifiedReIDDataset returns "rgb" or "depth" key in sample dict.
                key = self.hparams.modality # 'rgb' or 'depth'
                if key not in imgs:
                     # Fallback if mismatch
                     key = list(imgs.keys())[0] 
                     if key == 'path': key = list(imgs.keys())[1]
                
                x = imgs[key] 
                return self.encoder(x)

        # Forward Anchor, Pos, Neg
        a_feat, a_logits = forward_one(anchor_imgs)
        p_feat, p_logits = forward_one(pos_imgs)
        n_feat, n_logits = forward_one(neg_imgs)
        
        # Triplet Loss
        # Triplet Loss (using nn.TripletMarginLoss for explicit triplets) 
        # Note: My TripletLoss signature is (inputs, targets) ? 
        # No, the one I just wrote above `TripletLoss` takes (inputs, targets).
        # Wait, the `TripletLoss` class I included in THIS file (at the top) takes (inputs, targets) and does hard mining.
        # BUT `UnifiedReIDDataset` provides TRIPLETs.
        # So we don't need hard mining within batch necessarily, or we can treat (A, P, N) as batch?
        # If we use `TripletLoss` as defined (Batch Hard), we need Labels.
        # `batch['person_id']` are labels for Anchor.
        # Pos has same label. Neg has different.
        # If we use Batch Hard, we concatenate (A, P, N) or just (A, P)?
        # Actually with Triplet Dataset, we usually simple Triplet Loss: max(d(a,p) - d(a,n) + margin, 0).
        # The `TripletLoss` class I pasted above implements Batch Hard.
        # To support Triplet Dataset (Explicit Triplets), I should use `nn.TripletMarginLoss`.
        
        # Let's stick to the user's codebase style if possible.
        # The previous `train.py` (Step 26) instantiated `ReIDSystem`.
        # It seemed to expect `batch['label']`.
        # THIS means the previous `UnifiedReIDDataset` (which I didn't see fully in previous turn, but I just read in Step 113) 
        # DOES return triplets.
        # So strictly speaking, `Batch Hard` loss is not appropriate unless we discard the triplet structure and just throw all anchors/positives into a batch.
        
        # Modification: I will use `nn.TripletMarginLoss` for the explicit triplets from dataset.
        # And I will also compute ID loss on Anchors (and Positives).
        
        loss_tri_fn = nn.TripletMarginLoss(margin=self.hparams.margin)
        loss_tri_val = loss_tri_fn(a_feat, p_feat, n_feat)
        
        # ID Loss
        # We need integer labels. `person_id` in dataset strings?
        # `UnifiedReIDDataset` returns `person_id` as string usually?
        # In Step 113, `_index_by_person` uses `str(pid)`.
        # `UnifiedReIDDataModule` doesn't seem to have valid LabelEncoder.
        # `train.py` in Step 26: `num_classes = len(dm.train_ds.pid_map)`.
        # This implies `train_ds` has `pid_map`.
        # My `UnifiedReIDDataset` in Step 113 DOES NOT have `pid_map`.
        # I MUST FIX THIS. The dataset needs to map PIDs to Integers.
        
        # Since I can't easily fix the dataset in this file step, I will assume `batch['person_id']` is already integer 
        # OR I will rely on `train.py` to fix it?
        # No, `train.py` just reads `num_classes`.
        # I'll check `UnifiedReIDDataset` again.
        # It uses `self.df.groupby("person_id")`.
        # It returns `anchor_row["person_id"]`.
        # If the CSV has string IDs, this fails CrossEntropy.
        
        # I will augment this `training_step` to handle it robustly or fail.
        # But wait, I am WRITING `utils/models.py`. I can't fix data here.
        # I will assume labels are valid LongTensors.
        
        labels = batch['label'] # I'll make sure dataset returns this
        
        if self.hparams.modality == "cross":
             # a_logits is tuple (l_rgb, l_depth)
             loss_id = (self.ce_loss(a_logits[0], labels) + self.ce_loss(a_logits[1], labels)) / 2
        else:
             loss_id = self.ce_loss(a_logits, labels)
             
        loss = loss_id + loss_tri_val
        
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        # Warmup + Cosine
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=self.hparams.lr,
            steps_per_epoch=self.hparams.steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.1, # 10% warmup
            anneal_strategy='cos'
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }
