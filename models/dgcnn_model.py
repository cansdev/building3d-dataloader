#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# Basic DGCNN model for Building3D wireframe reconstruction

import torch
import torch.nn as nn
import torch.nn.functional as F

def knn_graph(x, k=20):
    """
    Build k-NN graph from point features.
    
    Args:
        x: [B, N, D] point features
        k: number of nearest neighbors
    
    Returns:
        idx: [B, N, k] k-NN indices
    """
    B, N, D = x.shape
    
    # Compute pairwise distances
    inner = -2 * torch.matmul(x, x.transpose(2, 1))  # [B, N, N]
    xx = torch.sum(x**2, dim=2, keepdim=True)  # [B, N, 1]
    dist = xx + inner + xx.transpose(2, 1)  # [B, N, N]
    
    # Get k nearest neighbors
    _, idx = torch.topk(-dist, k=k+1, dim=-1)  # [B, N, k+1]
    idx = idx[:, :, 1:]  # Remove self-connection [B, N, k]
    
    return idx

def get_edge_features(x, idx):
    """
    Extract edge features for DGCNN edge convolution.
    
    Args:
        x: [B, N, D] point features  
        idx: [B, N, k] k-NN indices
    
    Returns:
        edge_features: [B, N, k, 2*D] concatenated edge features
    """
    B, N, D = x.shape
    k = idx.shape[2]
    
    # Get neighbor features
    batch_idx = torch.arange(B).view(-1, 1, 1).expand(B, N, k).to(x.device)
    neighbors = x[batch_idx, idx]  # [B, N, k, D]
    
    # Concatenate center and neighbor features
    center = x.unsqueeze(2).expand(B, N, k, D)  # [B, N, k, D]
    edge_features = torch.cat([center, neighbors - center], dim=-1)  # [B, N, k, 2*D]
    
    return edge_features

class EdgeConv(nn.Module):
    """
    Basic Edge Convolution layer for DGCNN.
    """
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, C] point features
        Returns:
            [B, N, out_channels] updated features
        """
        B, N, C = x.shape
        
        # Build k-NN graph
        idx = knn_graph(x, self.k)  # [B, N, k]
        
        # Get edge features
        edge_feat = get_edge_features(x, idx)  # [B, N, k, 2*C]
        
        # Reshape for conv2d: [B, 2*C, N, k]
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()
        
        # Apply edge convolution
        edge_feat = self.conv(edge_feat)  # [B, out_channels, N, k]
        
        # Max pooling over neighbors
        x = torch.max(edge_feat, dim=-1)[0]  # [B, out_channels, N]
        
        return x.transpose(2, 1).contiguous()  # [B, N, out_channels]

class BasicDGCNN(nn.Module):
    """
    Efficient DGCNN model for Building3D wireframe reconstruction.
    Input: XYZ + GroupID + BorderWeight (5 features)
    Output: Direct vertex coordinate prediction
    Optimized for lower parameter count while maintaining performance.
    """
    def __init__(self, input_dim=5, k=20, max_vertices=64):
        super().__init__()
        
        self.k = k
        self.input_dim = input_dim
        self.max_vertices = max_vertices
        
        # Reduced edge convolution layers - smaller channels but same depth
        self.edge_conv1 = EdgeConv(input_dim, 32, k)   # 5→32 (was 64)
        self.edge_conv2 = EdgeConv(32, 64, k)          # 32→64 (was 128)
        self.edge_conv3 = EdgeConv(64, 128, k)         # 64→128 (was 256)
        
        # Smaller global feature aggregation
        self.conv_global = nn.Sequential(
            nn.Conv1d(32 + 64 + 128, 256, kernel_size=1, bias=False),  # 224→256 (was 448→512)
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Global max pooling to get scene-level features
        # Output: [B, 256] scene features (was 512)
        
        # Compact vertex coordinate prediction head
        self.vertex_predictor = nn.Sequential(
            nn.Linear(256, 512),           # 256→512 (was 512→1024)
            nn.ReLU(),
            nn.Dropout(0.2),               # Reduced dropout
            nn.Linear(512, max_vertices * 3)  # Direct to output (removed middle layer)
        )
        
        # Simplified number of vertices prediction head
        self.num_vertices_predictor = nn.Sequential(
            nn.Linear(256, 64),            # 256→64 (was 512→256)
            nn.ReLU(),
            nn.Linear(64, 1),              # 64→1 (was 256→64→1)
            nn.Sigmoid()  # Ensure 0-1 range
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] input features
        
        Returns:
            dict with vertex_coords, num_vertices, global_features
        """
        B, N, _ = x.shape
        
        # Extract edge convolution features (smaller channels)
        x1 = self.edge_conv1(x)      # [B, N, 32]  (was 64)
        x2 = self.edge_conv2(x1)     # [B, N, 64]  (was 128)
        x3 = self.edge_conv3(x2)     # [B, N, 128] (was 256)
        
        # Concatenate multi-scale features
        x_concat = torch.cat([x1, x2, x3], dim=2)  # [B, N, 224] (was 448)
        
        # Global features
        x_global = x_concat.transpose(2, 1)  # [B, 224, N]
        x_global = self.conv_global(x_global)  # [B, 256, N] (was 512)
        
        # Global max pooling to get scene-level representation
        global_features = torch.max(x_global, dim=2)[0]  # [B, 256] (was 512)
        
        # Predict vertex coordinates
        vertex_coords = self.vertex_predictor(global_features)  # [B, max_vertices * 3]
        vertex_coords = vertex_coords.view(B, self.max_vertices, 3)  # [B, max_vertices, 3]
        
        # Predict number of vertices - Simplified architecture
        num_vertices_sigmoid = self.num_vertices_predictor(global_features)  # [B, 1] already sigmoid
        num_vertices = torch.clamp(torch.round(num_vertices_sigmoid * self.max_vertices), 
                                   min=1, max=self.max_vertices).long()  # [B, 1]
        
        return {
            'vertex_coords': vertex_coords,
            'num_vertices': num_vertices.squeeze(-1),  # [B]
            'num_vertices_sigmoid': num_vertices_sigmoid.squeeze(-1),  # [B] - for loss calculation
            'global_features': global_features,
            'point_features': x_concat
        }