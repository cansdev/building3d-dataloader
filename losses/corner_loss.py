import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in corner detection
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, N] - predicted corner logits
            targets: [B, N] - binary corner labels (0 or 1)
        """
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DistanceWeightedLoss(nn.Module):
    """
    Distance-weighted loss that gives higher weight to points closer to ground truth corners
    """
    def __init__(self, base_loss, distance_threshold=0.1, weight_factor=2.0):
        super(DistanceWeightedLoss, self).__init__()
        self.base_loss = base_loss
        self.distance_threshold = distance_threshold
        self.weight_factor = weight_factor

    def forward(self, inputs, targets, point_coords, corner_coords):
        """
        Args:
            inputs: [B, N] - predicted corner logits
            targets: [B, N] - binary corner labels
            point_coords: [B, N, C] - point cloud coordinates (C can be > 3)
            corner_coords: [B, M, 3] - ground truth corner coordinates
        """
        # Compute base loss per-point (without reduction)
        base_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # [B, N]
        
        # Apply focal weighting
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.base_loss.alpha * (1 - p_t) ** self.base_loss.gamma
        base_loss = focal_weight * base_loss  # [B, N]
        
        # Compute distance weights
        B, N = inputs.shape
        weights = torch.ones_like(inputs)
        
        for b in range(B):
            # Get valid corners (not padded)
            valid_mask = corner_coords[b, :, 0] > -1e0
            valid_corners = corner_coords[b, valid_mask]  # [M_valid, 3]
            
            if len(valid_corners) == 0:
                # No valid corners, keep default weights (1.0)
                continue
                
            # Compute distances from each point to all corners
            points = point_coords[b, :, :3]  # Extract only XYZ coordinates [N, 3]
            distances = torch.cdist(points, valid_corners, p=2)  # [N, M_valid]
            min_distances, _ = torch.min(distances, dim=1)  # [N]
            
            # Create distance-based weights
            # Points closer to corners get higher weights
            distance_weights = torch.exp(-min_distances / self.distance_threshold)
            weights[b, :] = 1.0 + self.weight_factor * distance_weights
        
        # Apply weights to loss
        weighted_loss = base_loss * weights
        
        return weighted_loss.mean()

class AdaptiveCornerLoss(nn.Module):
    """
    Adaptive corner loss that adjusts focal gamma based on training progress
    """
    def __init__(self, initial_focal_gamma=2.0, min_focal_gamma=0.5, 
                 distance_threshold=0.1, distance_weight=1.0, weight_factor=2.0):
        super(AdaptiveCornerLoss, self).__init__()
        
        self.initial_focal_gamma = initial_focal_gamma
        self.min_focal_gamma = min_focal_gamma
        self.current_gamma = initial_focal_gamma
        self.epoch = 0
        
        # Base focal loss with adaptive gamma
        self.focal_loss = FocalLoss(alpha=1.0, gamma=self.current_gamma)
        
        # Distance-weighted loss
        self.distance_loss = DistanceWeightedLoss(
            base_loss=self.focal_loss,
            distance_threshold=distance_threshold,
            weight_factor=weight_factor
        )
        
        self.distance_weight = distance_weight

    def update_epoch(self, epoch):
        """Update focal gamma based on training progress"""
        self.epoch = epoch
        
        # Gradually reduce gamma from initial to minimum
        # This helps with class imbalance early in training
        progress = min(epoch / 100.0, 1.0)  # Full transition over 100 epochs
        self.current_gamma = self.initial_focal_gamma - progress * (self.initial_focal_gamma - self.min_focal_gamma)
        
        # Update focal loss with new gamma
        self.focal_loss.gamma = self.current_gamma

    def forward(self, inputs, targets, point_coords, corner_coords):
        """
        Args:
            inputs: [B, N] - predicted corner logits
            targets: [B, N] - binary corner labels
            point_coords: [B, N, C] - point cloud coordinates (C can be > 3)
            corner_coords: [B, M, 3] - ground truth corner coordinates
        """
        # Compute focal loss with current gamma
        focal_loss = self.focal_loss(inputs, targets)
        
        # Compute distance-weighted loss
        distance_loss = self.distance_loss(inputs, targets, point_coords, corner_coords)
        
        # Combine losses
        total_loss = focal_loss + self.distance_weight * distance_loss
        
        return {
            'total_loss': total_loss,
            'focal_loss': focal_loss,
            'distance_loss': distance_loss
        }


def create_corner_labels_improved(point_clouds, wf_vertices, distance_threshold=0.05, soft_labels=True):
    """
    Create improved corner labels with soft labeling and better distance handling
    
    Args:
        point_clouds: [B, N, C] - point cloud data
        wf_vertices: [B, M, 3] - wireframe vertices (corners)
        distance_threshold: maximum distance to consider a point as a corner
        soft_labels: if True, use soft labels based on distance; if False, use binary labels
    
    Returns:
        corner_labels: [B, N] - corner labels (binary or soft)
    """
    B, N, C = point_clouds.shape
    corner_labels = torch.zeros(B, N, device=point_clouds.device)
    
    # Extract XYZ coordinates from point clouds
    pc_xyz = point_clouds[:, :, :3]  # [B, N, 3]
    
    for b in range(B):
        # Get valid wireframe vertices (not padded with -1e1)
        valid_mask = wf_vertices[b, :, 0] > -1e0  # Check if not padded
        valid_wf_vertices = wf_vertices[b, valid_mask]  # [M_valid, 3]
        
        if len(valid_wf_vertices) == 0:
            continue  # No valid wireframe vertices for this batch
            
        # Compute distances from each point cloud point to all wireframe vertices
        pc_points = pc_xyz[b]  # [N, 3]
        wf_points = valid_wf_vertices  # [M_valid, 3]
        
        # Compute pairwise distances: [N, M_valid]
        distances = torch.cdist(pc_points, wf_points, p=2)
        
        # Find minimum distance for each point cloud point
        min_distances, _ = torch.min(distances, dim=1)  # [N]
        
        if soft_labels:
            # Create soft labels based on distance
            # Points closer to corners get higher values
            corner_labels[b] = torch.exp(-min_distances / distance_threshold)
            # Clamp to [0, 1] range
            corner_labels[b] = torch.clamp(corner_labels[b], 0.0, 1.0)
        else:
            # Binary labels
            corner_mask = min_distances < distance_threshold
            corner_labels[b, corner_mask] = 1.0
    
    return corner_labels

