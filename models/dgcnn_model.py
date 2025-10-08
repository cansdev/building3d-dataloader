#!/usr/bin/python3
# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(x, k):
    """Fast k-NN using PyTorch ops."""
    inner = -2 * torch.matmul(x, x.transpose(2, 1))
    xx = torch.sum(x**2, dim=2, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

def get_graph_feature(x, k=20, idx=None):
    """Edge feature construction with geometric features.
    
    Input x: [B, N, C] where first 3 dims are xyz
    Returns: [B, N, k, feature_dim] edge features with geometric info
    """
    B, N, C = x.shape
    
    if idx is None:
        # KNN using xyz coordinates
        xyz = x[:, :, :3]
        idx = knn(xyz, k=k)
    
    batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
    neighbors = x[batch_idx, idx, :]
    x_expanded = x.unsqueeze(2).expand(B, N, k, C)
    
    # Base edge features: [center, neighbor - center, neighbor]
    edge_features = torch.cat([
        x_expanded,                    # Center point features
        neighbors - x_expanded,        # Relative features
        neighbors                      # Neighbor features
    ], dim=3)  # Total: C * 3 dimensions
    

    # Add geometric features (low noise, high signal)
    xyz_center = x[:, :, :3]  # [B, N, 3]
    xyz_neighbors = neighbors[:, :, :, :3]  # [B, N, k, 3]
        
    # 1. Euclidean distance to each neighbor
    distances = torch.norm(xyz_neighbors - xyz_center.unsqueeze(2), dim=3, keepdim=True)  # [B, N, k, 1]
        
    # 2. Local density: mean distance to k neighbors (expanded to per-edge)
    local_density = distances.mean(dim=2, keepdim=True).expand(B, N, k, 1)  # [B, N, k, 1]
        
    # 3. Relative height: z-offset from local neighborhood mean
    z_center = xyz_center[:, :, 2:3]  # [B, N, 1]
    z_neighbors = xyz_neighbors[:, :, :, 2:3]  # [B, N, k, 1]
    z_local_mean = z_neighbors.mean(dim=2, keepdim=True)  # [B, N, 1, 1]
    relative_height = (z_center.unsqueeze(2) - z_local_mean).expand(B, N, k, 1)  # [B, N, k, 1]
    
    # 4. LOCAL PLANARITY (variance of distances (flat surface = low variance, corner = high variance)
    dist_variance = torch.var(distances, dim=2, keepdim=True).expand(B, N, k, 1)  # [B, N, k, 1]

    # 5. ANGLE DIFFERENCE (Compute angle between center→neighbor vector and mean direction)
    vectors = xyz_neighbors - xyz_center.unsqueeze(2)  # [B, N, k, 3] vectors from center to neighbors
    vectors_norm = vectors / (torch.norm(vectors, dim=3, keepdim=True) + 1e-6)  # Normalize
    mean_direction = vectors_norm.mean(dim=2, keepdim=True)  # [B, N, 1, 3]
    mean_direction = mean_direction / (torch.norm(mean_direction, dim=3, keepdim=True) + 1e-6) 
    cos_angles = (vectors_norm * mean_direction).sum(dim=3, keepdim=True)  # [B, N, k, 1]
    angle_score = 1.0 - torch.abs(cos_angles)  # High value = sharp angle = corner
        
    # Concatenate geometric features: +5 dimensions per edge
    edge_features = torch.cat([
        edge_features,      # Original features (C * 3)
        distances,          
        local_density,      
        relative_height,    
        dist_variance,      
        angle_score
    ], dim=3)  # Total: C*3 + 5 dimensions
    
    return edge_features


class EdgeConv(nn.Module):
    """EdgeConv with geometric features."""
    
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        
        # Standard 3x edge features + 5 geometric features (always enabled)
        edge_feat_dim = in_channels * 3 + 5  # distance, local_density, relative_height, planarity, corner_angle
        
        self.conv = nn.Sequential(
            nn.Conv2d(edge_feat_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        x_t = x.transpose(2, 1)
        edge_feat = get_graph_feature(x_t, k=self.k)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = self.conv(edge_feat)
        return out.max(dim=-1)[0]

class BasicDGCNN(nn.Module):
    """
    Simple DGCNN for vertex prediction.
    
    Input: [B, N, 5] where dims are [x, y, z, groupid, borderweight]
    Output: Direct vertex coordinates (absolute positions)
    """
    def __init__(self, input_dim=5, k=20, max_vertices=64):
        super().__init__()
        self.k = k
        self.max_vertices = max_vertices
        
        # 3-layer EdgeConv hierarchy with geometric features (always enabled)
        self.edge_conv1 = EdgeConv(5, 48, k=k)
        self.edge_conv2 = EdgeConv(48, 96, k=k)
        self.edge_conv3 = EdgeConv(96, 128, k=k)

        # Skip connections
        self.skip1 = nn.Conv1d(48, 96, 1, bias=False)    # 4,608
        self.skip2 = nn.Conv1d(96, 128, 1, bias=False)   # 12,288
        
        # Global pooling: 128 max + 128 avg = 256
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 192),              # 49,152 + 192
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(192, max_vertices * 4)  # 12,288 + 256
        )
        
        # Init existence bias
        with torch.no_grad():
            self.decoder[-1].bias[3::4] = -1.0
        
    def forward(self, x, return_absolute=True):
        """
        Forward pass - predicts absolute vertex coordinates.
        
        Args:
            x: [B, N, 5] input point cloud [xyz, groupid, borderweight]
            return_absolute: Kept for compatibility, always returns absolute coords
        
        Returns:
            dict with vertex predictions
        """
        B, N, _ = x.shape
        
        # EdgeConv hierarchy
        x_feat = x.transpose(1, 2)  # [B, 5, N]
        
        f1 = self.edge_conv1(x_feat)                  # [B, 48, N]
        f2 = self.edge_conv2(f1) + self.skip1(f1)     # [B, 96, N]
        f3 = self.edge_conv3(f2) + self.skip2(f2)     # [B, 128, N]
        
        # Global features
        glob = torch.cat([
            f3.max(dim=2)[0],
            f3.mean(dim=2)
        ], dim=1)  # [B, 256]
        
        # Decode to absolute coordinates
        out = self.decoder(glob).view(B, self.max_vertices, 4)
        
        coords = out[:, :, :3]   # [B, max_vertices, 3] - absolute coordinates
        logits = out[:, :, 3]    # [B, max_vertices] - existence logits
        probs = torch.sigmoid(logits)
        
        return {
            'vertex_coords': coords,           # Absolute coordinates
            'vertex_offsets': coords,          # Same as coords (for compatibility)
            'pc_centroid': torch.zeros(B, 1, 3, device=x.device),  # Dummy for compatibility
            'existence_logits': logits,
            'existence_probs': probs,
            'num_vertices': (probs > 0.5).sum(dim=1).clamp(1, self.max_vertices),
            'global_features': glob,
        }

