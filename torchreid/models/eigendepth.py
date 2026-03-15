from __future__ import division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from .osnet import osnet_x1_0

def compute_normals_and_coords(depth_map):
    """
    Compute point cloud (x, y, z) and normals (nx, ny, nz) from depth map.
    Assumes depth_map is [B, 1, H, W] normalized or raw.
    """
    B, C, H, W = depth_map.size()
    
    # Coordinate grid (x, y)
    y_grid, x_grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    x_grid = x_grid.to(depth_map.device).unsqueeze(0).expand(B, -1, -1).float()
    y_grid = y_grid.to(depth_map.device).unsqueeze(0).expand(B, -1, -1).float()
    
    # z is depth
    z = depth_map.squeeze(1) # [B, H, W]
    
    # Gradients for normals (approximate)
    # dz/dx, dz/dy
    dz_dx = z[:, :, 1:] - z[:, :, :-1]
    dz_dy = z[:, 1:, :] - z[:, :-1, :]
    
    # Pad to original size
    dz_dx = F.pad(dz_dx, (0, 1, 0, 0))
    dz_dy = F.pad(dz_dy, (0, 0, 0, 1))
    
    # Normal vector (-dz/dx, -dz/dy, 1) normalized
    n_x = -dz_dx
    n_y = -dz_dy
    n_z = torch.ones_like(z)
    
    norm = torch.sqrt(n_x**2 + n_y**2 + n_z**2 + 1e-6)
    n_x = n_x / norm
    n_y = n_y / norm
    n_z = n_z / norm
    
    # Concatenate features: [B, 6, H, W] -> (x, y, z, nx, ny, nz)
    # Normalize coordinates to resemble similar scale
    x_grid = x_grid / W
    y_grid = y_grid / H 
    # z is already normalized (assuming input is tensor)
    
    features = torch.stack([x_grid, y_grid, z, n_x, n_y, n_z], dim=1)
    return features

def compute_local_covariance(features, grid_h=6, grid_w=2):
    """
    Splits features into grid_h x grid_w voxels and computes covariance.
    features: [B, 6, H, W]
    Returns: [B, num_voxels, 6, 6]
    """
    B, C, H, W = features.size()
    
    # Patch size
    ph = H // grid_h
    pw = W // grid_w
    
    cov_matrices = []
    
    for i in range(grid_h):
        for j in range(grid_w):
            # Extract voxel
            voxel = features[:, :, i*ph:(i+1)*ph, j*pw:(j+1)*pw] # [B, 6, ph, pw]
            # Flatten spatial dimensions
            voxel = voxel.flatten(2) # [B, 6, N_points]
            
            # Center data
            mean = voxel.mean(dim=2, keepdim=True)
            voxel_centered = voxel - mean
            
            # Compute Covariance: (X * X^T) / (N-1)
            N = voxel.size(2)
            if N > 1:
                cov = torch.bmm(voxel_centered, voxel_centered.transpose(1, 2)) / (N - 1)
            else:
                cov = torch.eye(6).to(features.device).expand(B, -1, -1) * 1e-6
            
            # Regularize (add epsilon to diagonal to ensure positive definite)
            eps = 1e-5 * torch.eye(6).to(features.device).expand(B, -1, -1)
            cov = cov + eps
            
            cov_matrices.append(cov)
            
    return torch.stack(cov_matrices, dim=1) # [B, num_voxels, 6, 6]

def compute_log_eigenvalues(cov_matrices):
    """
    Computes Log-Eigenvalues of covariance matrices.
    cov_matrices: [B, num_voxels, 6, 6]
    Returns: [B, num_voxels * 6]
    """
    # Eigenvalues
    # torch.linalg.eigvalsh is for symmetric (Hermitian) matrices
    eigs = torch.linalg.eigvalsh(cov_matrices) # [B, num_voxels, 6]
    
    # Logarithm
    log_eigs = torch.log(eigs + 1e-6)
    
    # Flatten
    return log_eigs.flatten(1) # [B, num_voxels * 6]

class EigenDepth(nn.Module):
    """
    Implementation of Eigen-Depth Feature (Wu et al. 2017).
    """
    def __init__(self, num_classes, loss='softmax', grid_h=6, grid_w=2, **kwargs):
        super(EigenDepth, self).__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.feature_dim = grid_h * grid_w * 6
        
        # Classifier layer (Linear)
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.loss = loss
        
    def forward(self, x):
        # x: [B, 1, H, W] or [B, 3, H, W] (take 1st channel if 3)
        if x.size(1) == 3:
            x = x[:, 0:1, :, :]
            
        features_map = compute_normals_and_coords(x) # [B, 6, H, W]
        covs = compute_local_covariance(features_map, self.grid_h, self.grid_w) # [B, V, 6, 6]
        eigen_feats = compute_log_eigenvalues(covs) # [B, V*6]
        
        if not self.training:
            return eigen_feats
            
        y = self.classifier(eigen_feats)
        
        if self.loss == {'softmax'}:
            return y
        elif self.loss == 'triplet':
            return y, eigen_feats
        else:
             return y

class CrossModalEigenDepth(nn.Module):
    """
    Two-Stream Networks:
    1. RGB Stream: OSNet
    2. Depth Stream: EigenDepth
    
    Maps RGB to EigenDepth space (Implicit Feature Transfer).
    """
    def __init__(self, num_classes, loss='softmax', **kwargs):
        super(CrossModalEigenDepth, self).__init__()
        
        # RGB Stream: Initialize OSNet with 'triplet' loss configuration 
        # to ensure it returns features during training.
        # We will use the backbone features directly.
        self.rgb_model = osnet_x1_0(num_classes=num_classes, pretrained=True, loss='triplet')
        self.rgb_dim = self.rgb_model.feature_dim
        
        # Depth Stream
        self.depth_model = EigenDepth(num_classes, loss=loss, grid_h=6, grid_w=2)
        self.depth_dim = self.depth_model.feature_dim
        
        # Mapping: RGB (512) -> Depth (72)
        self.rgb_projector = nn.Sequential(
            nn.Linear(self.rgb_dim, self.depth_dim),
            nn.BatchNorm1d(self.depth_dim),
            nn.ReLU(inplace=True)
        )
        
        # Shared Classifier
        self.classifier = nn.Linear(self.depth_dim, num_classes)
        self.loss = loss

    def forward(self, x):
        """
        Forward pass with automatic modality detection.
        """
        # Heuristic to detect if input is Depth (Grayscale/Replicated 3-channel) or RGB (Color).
        # We check the difference between channels.
        # If mean difference is very small, it's Depth.
        
        # x: [B, 3, H, W]
        with torch.no_grad():
             diff = torch.abs(x[:, 0] - x[:, 1]).mean() + torch.abs(x[:, 1] - x[:, 2]).mean()
             is_depth = (diff < 1e-3)
             
        if is_depth:
            # Depth Path
            # Use only 1 channel
            x_d = x[:, 0:1, :, :]
            features = self.depth_model(x_d) # [B, feat_dim]
        else:
            # RGB Path
            backbone_out = self.rgb_model(x)
            
            # Extract features from OSNet output
            if isinstance(backbone_out, tuple):
                 rgb_feats = backbone_out[1] # (logits, features) for triplet+softmax
            else:
                 rgb_feats = backbone_out # Just features
            
            # Project to EigenDepth dimension
            features = self.rgb_projector(rgb_feats)
        
        # Classification
        if not self.training:
            return features
            
        y = self.classifier(features)
        
        if self.loss == {'softmax'}:
            return y
        elif self.loss == 'triplet':
            return y, features
        else:
            return y
