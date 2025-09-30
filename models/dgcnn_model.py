#!/usr/bin/python3
# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_pca_alignment(coords, return_transform=True):
    """
    Compute PCA-based canonical orientation for point cloud with CONSISTENT orientation.
    This makes the model rotation-invariant by aligning to principal axes.
    
    CRITICAL FIX: Enforce deterministic eigenvector directions to avoid sign ambiguity!
    - Each axis points toward the side with MORE mass (more points)
    - This ensures the same canonical orientation regardless of input rotation
    
    Args:
        coords: [B, N, 3] point coordinates
        return_transform: if True, return rotation matrix
    
    Returns:
        aligned_coords: [B, N, 3] coordinates aligned to principal axes
        pca_features: [B, 9] rotation-invariant features (eigenvalues + extents)
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
        
        # ===== CRITICAL FIX: DETERMINISTIC EIGENVECTOR ORIENTATION =====
        # Flip each eigenvector so it points toward the "heavier" side
        # This resolves the +/- sign ambiguity in PCA!
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
        pca_features = torch.cat([
            eigen_normalized,  # 3 - normalized eigenvalues (shape descriptor)
            extents_size,      # 3 - extents along principal axes
            centroid.squeeze(1) # 3 - center position (this WILL rotate, but needed for position)
        ], dim=1)  # [B, 9]
        
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
            centroid.squeeze(1)                    # centroid
        ], dim=1)
        if return_transform:
            return centered, pca_features, identity, centroid
        else:
            return centered, pca_features


class BasicDGCNN(nn.Module):
    """
    SIMPLE point cloud to vertices model.
    No edge convolutions, no attention, no complex operations.
    Just: Point features → Global pooling → MLP → Vertex predictions
    
    Input: [B, N, 5] - XYZ + GroupID + BorderWeight
    Output: [B, max_vertices, 4] - XYZ + existence per vertex
    
    Architecture:
    1. Simple per-point MLPs to extract features
    2. Max pooling to get global features
    3. MLP decoder to predict all vertices
    """
    def __init__(self, input_dim=5, k=20, max_vertices=64):
        super().__init__()
        
        self.input_dim = input_dim
        self.max_vertices = max_vertices
        
        # Simple point-wise feature extraction (no neighbors, no graphs)
        # Process ONLY semantic features (GroupID + BorderWeight), NOT coordinates!
        # This prevents rotation-invariant feature learning
        self.point_mlp = nn.Sequential(
            nn.Linear(2, 32),  # Input: GroupID + BorderWeight (NOT XYZ!)
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        
        # Global aggregation (max pooling across all points)
        # 201 global features (192 learned + 9 PCA-based rotation-invariant features)
        
        # Vertex decoder - predict all vertices from global features
        self.vertex_decoder = nn.Sequential(
            nn.Linear(201, 256),  # Input: 201 features (192 learned + 9 PCA features)
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, max_vertices * 4)  # 4 outputs per vertex (x,y,z,existence)
        )
        
        # Initialize biases
        # Set existence logits to slightly negative (start by predicting "not exist")
        with torch.no_grad():
            final_layer = self.vertex_decoder[-1]
            for i in range(max_vertices):
                final_layer.bias[i * 4 + 3] = -2.0  # Existence logit starts negative
        
    def forward(self, x):
        """
        Forward pass with PCA-based rotation-invariant learning.
        
        The key insight: Learn shape in canonical (PCA-aligned) space,
        then transform predictions back to original orientation.
        
        Args:
            x: [B, N, D] input point features (B=batch, N=points, D=features)
               D=5: XYZ coordinates + GroupID + BorderWeight
        
        Returns:
            dict with predictions in ORIGINAL orientation
        """
        B, N, D = x.shape
        
        # CRITICAL FIX: Keep XYZ separate from learned features!
        coords = x[:, :, :3]  # [B, N, 3] - Raw XYZ coordinates
        semantic_features = x[:, :, 3:]  # [B, N, 2] - GroupID + BorderWeight
        
        # ===== ROTATION-INVARIANT PROCESSING =====
        # Step 1: Align point cloud to canonical orientation (PCA axes)
        aligned_coords, pca_features, rotation_matrix, centroid = compute_pca_alignment(
            coords, return_transform=True
        )
        # aligned_coords: [B, N, 3] - coordinates in canonical space
        # pca_features: [B, 9] - rotation-invariant shape descriptors
        # rotation_matrix: [B, 3, 3] - rotation from original to canonical
        # centroid: [B, 1, 3] - center of point cloud
        
        # Step 2: Extract features ONLY from semantic info (NOT coordinates!)
        point_features = self.point_mlp(semantic_features)  # [B, N, 64]
        
        # Step 3: Aggregate learned features (rotation-invariant pooling)
        max_features = torch.max(point_features, dim=1)[0]  # [B, 64]
        avg_features = torch.mean(point_features, dim=1)     # [B, 64]
        std_features = torch.std(point_features, dim=1)      # [B, 64]
        
        # Step 4: Combine rotation-invariant features
        # - Learned features (64+64+64 = 192): describe structure
        # - PCA features (9): describe shape & orientation
        global_features = torch.cat([
            max_features,     # 64 - semantic structure (rotation-invariant)
            avg_features,     # 64 - feature distribution
            std_features,     # 64 - feature variance
            pca_features,     # 9 - eigenvalues (3) + extents (3) + centroid (3)
        ], dim=1)  # Total: 192 + 9 = 201 features
        
        # Step 5: Decode vertices in CANONICAL space
        vertex_output = self.vertex_decoder(global_features)  # [B, max_vertices * 4]
        vertex_output = vertex_output.view(B, self.max_vertices, 4)
        
        vertex_coords_canonical = vertex_output[:, :, :3]  # [B, max_vertices, 3]
        existence_logits = vertex_output[:, :, 3]           # [B, max_vertices]
        
        # ===== KEEP PREDICTIONS IN CANONICAL SPACE =====
        # Instead of rotating back to input orientation, we keep predictions in canonical space.
        # This ensures training and visualization both compare in the SAME coordinate system.

        # Logic:
        # - Training: GT is transformed to canonical → compare with canonical predictions
        # - Visualization: GT is transformed to canonical → compare with canonical predictions
        # - Augmentation: Different rotations → SAME canonical space (deterministic PCA)
        
        # Step 6: Existence probabilities
        existence_probs = torch.sigmoid(existence_logits)
        vertex_exists = existence_probs > 0.5
        num_vertices = vertex_exists.sum(dim=1)
        num_vertices = torch.clamp(num_vertices, min=1, max=self.max_vertices)
        
        return {
            'vertex_coords': vertex_coords_canonical,          # [B, max_vertices, 3] - PRIMARY OUTPUT (canonical space)
            'vertex_coords_canonical': vertex_coords_canonical, # [B, max_vertices, 3] - same as above
            'existence_logits': existence_logits,              # [B, max_vertices]
            'existence_probs': existence_probs,                # [B, max_vertices]
            'num_vertices': num_vertices,                      # [B]
            'global_features': global_features,                # [B, 201]
            'point_features': point_features,                  # [B, N, 64]
            'pca_features': pca_features,                      # [B, 9]
            'rotation_matrix': rotation_matrix,                # [B, 3, 3] - can be used to transform back if needed
            'centroid': centroid,                              # [B, 1, 3] - can be used to transform back if needed
            'aligned_coords': aligned_coords,                  # [B, N, 3] - for debugging
        }