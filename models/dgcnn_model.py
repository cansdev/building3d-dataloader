#!/usr/bin/python3
# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    """
    Find k-nearest neighbors for each point.
    
    Args:
        x: [B, N, C] - point features or coordinates
        k: int - number of nearest neighbors
    
    Returns:
        idx: [B, N, k] - indices of k nearest neighbors for each point
    """
    # Compute pairwise distance
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [B, N, N]
    xx = torch.sum(x**2, dim=2, keepdim=True)  # [B, N, 1]
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # [B, N, N] (negative for sorting)
    
    # Get k nearest neighbors (including self)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # [B, N, k]
    return idx


def get_graph_feature(x, k=20, idx=None):
    """
    Construct edge features for EdgeConv.
    
    For each point, gather its k nearest neighbors and compute edge features:
    edge_feature = [point_feature, neighbor_feature - point_feature]
    
    Args:
        x: [B, N, C] - point features
        k: int - number of nearest neighbors
        idx: [B, N, k] - precomputed neighbor indices (optional)
    
    Returns:
        edge_features: [B, N, k, 2*C] - features for each edge
    """
    B, N, C = x.shape
    device = x.device
    
    # Find k nearest neighbors if not provided
    if idx is None:
        idx = knn(x, k=k)  # [B, N, k]
    
    # Create batch indices for gathering
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, k)
    
    # Gather neighbor features
    neighbors = x[batch_idx, idx, :]  # [B, N, k, C]
    
    # Compute edge features
    x_expanded = x.unsqueeze(2).expand(B, N, k, C)  # [B, N, k, C]
    edge_features = torch.cat([
        x_expanded,                    # [B, N, k, C] - center point feature
        neighbors - x_expanded         # [B, N, k, C] - relative neighbor feature
    ], dim=3)  # [B, N, k, 2*C]
    
    return edge_features


def compute_pca_alignment(coords, return_transform=True):
    """
    Compute PCA-based canonical orientation for point cloud with consistent orientation.
    This makes the model rotation-invariant by aligning to principal axes.
    
    Enforce deterministic eigenvector directions to avoid sign ambiguity:
    - Each axis points toward the side with more mass (more points)
    - This ensures the same canonical orientation regardless of input rotation
    
    Args:
        coords: [B, N, 3] point coordinates
        return_transform: if True, return rotation matrix
    
    Returns:
        aligned_coords: [B, N, 3] coordinates aligned to principal axes
        pca_features: [B, 6] rotation-invariant features (eigenvalues + extents)
        rotation_matrix: [B, 3, 3] rotation to apply to predictions (optional)
    """
    B, N, _ = coords.shape
    device = coords.device
    
    # Center the point cloud
    centroid = coords.mean(dim=1, keepdim=True)  # [B, 1, 3]
    centered = coords - centroid  # [B, N, 3]
    
    # Compute covariance matrix for each batch
    # cov = (1/N) * X^T * X where X is centered coordinates
    cov = torch.bmm(centered.transpose(1, 2), centered) / N  # [B, 3, 3]
    
    # Compute eigenvalues and eigenvectors (PCA)
    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # eigenvalues sorted ascending
        
        # Sort in descending order (largest eigenvalue first)
        eigenvalues = torch.flip(eigenvalues, dims=[1])  # [B, 3]
        eigenvectors = torch.flip(eigenvectors, dims=[2])  # [B, 3, 3]
        
        # Ensure right-handed coordinate system (det = +1)
        # If determinant is negative, flip the last eigenvector
        det = torch.det(eigenvectors)  # [B]
        flip_mask = det < 0
        if flip_mask.any():
            eigenvectors[flip_mask, :, 2] *= -1
        
        # Deterministic eigenvector orientation
        # Flip each eigenvector so it points toward the "heavier" side
        for b in range(B):
            for axis in range(3):
                # Project points onto this eigenvector
                projections = torch.matmul(centered[b], eigenvectors[b, :, axis])  # [N]
                
                # Count points on positive vs negative side
                positive_mass = (projections > 0).sum().float()
                negative_mass = (projections < 0).sum().float()
                
                # If more mass on negative side, flip the eigenvector
                if negative_mass > positive_mass:
                    eigenvectors[b, :, axis] *= -1
        
        # Align coordinates to principal axes
        # This creates a CONSISTENT canonical orientation regardless of input rotation
        aligned = torch.bmm(centered, eigenvectors)  # [B, N, 3]
        
        # Compute rotation-invariant features
        # 1. Eigenvalues (describe spread along principal axes - rotation invariant!)
        eigen_normalized = eigenvalues / (eigenvalues.sum(dim=1, keepdim=True) + 1e-8)  # [B, 3]
        
        # 2. Extents along principal axes (bounding box in canonical space)
        extents_min = aligned.min(dim=1)[0]  # [B, 3]
        extents_max = aligned.max(dim=1)[0]  # [B, 3]
        extents_size = extents_max - extents_min  # [B, 3]
        
        # Combine into rotation-invariant feature vector
        # NOTE: Centroid removed - it's NOT rotation-invariant!
        pca_features = torch.cat([
            eigen_normalized,  # 3 - normalized eigenvalues (shape descriptor)
            extents_size,      # 3 - extents along principal axes
        ], dim=1)  # [B, 6]
        
        if return_transform:
            return aligned, pca_features, eigenvectors, centroid
        else:
            return aligned, pca_features
            
    except Exception as e:
        # Fallback if PCA fails: return original coords
        print(f"Warning: PCA failed ({e}), using original coordinates")
        identity = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        pca_features = torch.cat([
            torch.ones(B, 3, device=device) / 3,  # uniform eigenvalues
            torch.ones(B, 3, device=device),       # unit extents
        ], dim=1)  # [B, 6] - no centroid
        if return_transform:
            return centered, pca_features, identity, centroid
        else:
            return centered, pca_features


class EdgeConv(nn.Module):
    """
    Edge Convolution layer (core of DGCNN).
    
    For each point:
    1. Find k nearest neighbors in feature space
    2. Compute edge features: [point_feat, neighbor_feat - point_feat]
    3. Apply shared MLP to all edges
    4. Aggregate with max pooling
    
    This captures local geometric structure!
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        
        # Shared MLP for edge features (processes 2*in_channels because of concatenation)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        """
        Args:
            x: [B, C, N] - point features (channels first for conv)
        
        Returns:
            out: [B, C', N] - aggregated edge features
        """
        # Transpose to [B, N, C] for k-NN computation
        x_transposed = x.transpose(2, 1)  # [B, N, C]
        
        # Get edge features
        edge_features = get_graph_feature(x_transposed, k=self.k)  # [B, N, k, 2*C]
        
        # Permute to [B, 2*C, N, k] for Conv2d
        edge_features = edge_features.permute(0, 3, 1, 2)  # [B, 2*C, N, k]
        
        # Apply edge convolution
        edge_features = self.conv(edge_features)  # [B, C', N, k]
        
        # Aggregate across neighbors (max pooling)
        out = edge_features.max(dim=-1)[0]  # [B, C', N]
        
        return out


class BasicDGCNN(nn.Module):
    """
    Dynamic Graph CNN for point cloud to vertices prediction.
    
    TRUE DGCNN architecture with:
    - Multiple EdgeConv layers (dynamic graph construction)
    - k-NN in feature space (graphs rebuilt at each layer)
    - Local geometric feature learning
    - PCA alignment for rotation invariance
    
    Input: [B, N, 5] - XYZ + GroupID + BorderWeight
    Output: [B, max_vertices, 4] - XYZ + existence per vertex
    
    Architecture:
    1. EdgeConv layers (extract local + global features)
    2. Global aggregation (max + avg pooling)
    3. MLP decoder (predict all vertices)
    """
    def __init__(self, input_dim=5, k=20, max_vertices=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.k = k
        self.max_vertices = max_vertices
        
        # EdgeConv layers
        self.edge_conv1 = EdgeConv(5, 32, k=k)
        self.edge_conv2 = EdgeConv(32, 64, k=k)
        
        # Residual/shortcut connection for layer 2
        self.shortcut2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64)
        )
        
        # Optional: Additional point-wise feature transformation
        self.conv_post = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=1, bias=False),  # 32+64=96 concatenated features
            nn.BatchNorm1d(96),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        # Global aggregation
        # 198 global features (96 max + 96 avg + 6 PCA features)
        
        # Vertex decoder - SIMPLIFIED for small dataset
        self.vertex_decoder = nn.Sequential(
            nn.Linear(198, 128),                    # Reduced from 518→512
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(0.3),                         # Increased dropout
            nn.Linear(128, max_vertices * 4),        # Direct to output
            nn.Tanh()
        )
        
        # Initialize biases - balanced for learning
        # Bias -1.0 → sigmoid(-1) ≈ 0.27 (27% probability - reasonable for 32/64 occupancy)
        with torch.no_grad():
            final_layer = self.vertex_decoder[-2] 
            for i in range(max_vertices):
                final_layer.bias[i * 4 + 3] = -1.0  
        
    def forward(self, x):
        """
        Forward pass with DGCNN + PCA-based rotation invariance.
        
        Key features:
        - PCA alignment for canonical orientation
        - Dynamic graph construction at each EdgeConv layer
        - Hierarchical feature learning (local → global)
        - Multi-scale feature aggregation
        
        Args:
            x: [B, N, D] input point features (B=batch, N=points, D=features)
               D=5: XYZ coordinates + GroupID + BorderWeight
        
        Returns:
            dict with predictions in canonical orientation
        """
        B, N, D = x.shape
        
        # Separate coordinates from semantic features
        coords = x[:, :, :3]  # [B, N, 3] - Raw XYZ coordinates
        semantic_features = x[:, :, 3:]  # [B, N, 2] - GroupID + BorderWeight
        
        # ===== STEP 1: PCA ALIGNMENT (Rotation Invariance) =====
        aligned_coords, pca_features, rotation_matrix, centroid = compute_pca_alignment(
            coords, return_transform=True
        )
        # aligned_coords: [B, N, 3] - coordinates in canonical space
        # pca_features: [B, 6] - rotation-invariant shape descriptors
        
        # ===== STEP 2: DGCNN FEATURE EXTRACTION =====
        # Combine PCA-aligned coordinates with semantic features
        # This allows EdgeConv to see BOTH spatial structure AND semantic attributes
        combined_features = torch.cat([
            aligned_coords,      # [B, N, 3] - geometry in canonical space (rotation-invariant!)
            semantic_features    # [B, N, 2] - semantic attributes (GroupID + BorderWeight)
        ], dim=2)  # [B, N, 5]
        
        # Transpose to [B, C, N] for convolution operations
        features = combined_features.transpose(1, 2)  # [B, 5, N]
        
        # EdgeConv 1: Build graph in combined spatial+semantic space
        feat1 = self.edge_conv1(features)  # [B, 32, N]
        
        # EdgeConv 2: Build graph in learned 32-D feature space (dynamic!) + residual
        feat2 = self.edge_conv2(feat1) + self.shortcut2(feat1)  # [B, 64, N] with residual
        
        # ===== STEP 3: MULTI-SCALE FEATURE AGGREGATION =====
        # Concatenate features from both levels (skip connections)
        multi_scale_features = torch.cat([feat1, feat2], dim=1)  # [B, 96, N]
        
        # Apply additional transformation
        features_transformed = self.conv_post(multi_scale_features)  # [B, 96, N]
        
        # ===== STEP 4: GLOBAL AGGREGATION =====
        # Pool features across all points
        max_features = torch.max(features_transformed, dim=2)[0]  # [B, 96]
        avg_features = torch.mean(features_transformed, dim=2)     # [B, 96]
        
        # Combine with PCA features
        global_features = torch.cat([
            max_features,     # 96 - strongest local features
            avg_features,     # 96 - average structure
            pca_features,     # 6 - rotation-invariant shape descriptors
        ], dim=1)  # Total: 96 + 96 + 6 = 198 features
        
        # ===== STEP 5: VERTEX PREDICTION IN CANONICAL SPACE =====
        vertex_output = self.vertex_decoder(global_features)  # [B, max_vertices * 4]
        vertex_output = vertex_output.view(B, self.max_vertices, 4)
        
        vertex_coords_canonical = vertex_output[:, :, :3]  # [B, max_vertices, 3]
        existence_logits = vertex_output[:, :, 3]           # [B, max_vertices]
        
        # ===== STEP 6: EXISTENCE PROBABILITIES =====
        existence_probs = torch.sigmoid(existence_logits)
        vertex_exists = existence_probs > 0.5
        num_vertices = vertex_exists.sum(dim=1)
        num_vertices = torch.clamp(num_vertices, min=1, max=self.max_vertices)
        
        return {
            'vertex_coords': vertex_coords_canonical,          # [B, max_vertices, 3] - PRIMARY OUTPUT
            'vertex_coords_canonical': vertex_coords_canonical, # [B, max_vertices, 3] - same as above
            'existence_logits': existence_logits,              # [B, max_vertices]
            'existence_probs': existence_probs,                # [B, max_vertices]
            'num_vertices': num_vertices,                      # [B]
            'global_features': global_features,                # [B, 198]
            'multi_scale_features': features_transformed,      # [B, 96, N]
            'pca_features': pca_features,                      # [B, 6]
            'rotation_matrix': rotation_matrix,                # [B, 3, 3]
            'centroid': centroid,                              # [B, 1, 3]
            'aligned_coords': aligned_coords,                  # [B, N, 3] - for debugging
        }