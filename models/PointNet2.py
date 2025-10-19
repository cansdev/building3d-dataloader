import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation

class PointNet2CornerDetection(nn.Module):
    """
    PointNet2 for corner point detection with specific SA/FP architecture
    """
    def __init__(self, config):
        super(PointNet2CornerDetection, self).__init__()
        
        # Extract configuration
        self.input_channels = config.get('input_channels', 3)
        self.has_border_weights = config.get('use_border_weights', False)
        
        # Calculate feature channels (excluding XYZ and optional border weights)
        # Border weights are handled separately via attention if present
        feature_channels = self.input_channels - 3  # Remove XYZ
        if self.has_border_weights:
            feature_channels -= 1  # Remove border weights channel
        feature_channels = max(0, feature_channels)
        
        self.feature_channels = feature_channels
        
        # Create learnable border attention weight if border weights are used
        if self.has_border_weights:
            self.border_attention_weight = nn.Parameter(torch.tensor([0.0]))
            self.border_dropout = nn.Dropout(p=0.5)  # Dropout on border attention
            self.border_weights = None  # Will be set during forward pass
        else:
            self.border_attention_weight = None
            self.border_dropout = None
            self.border_weights = None
        
        print(f"    CornerNet init: total_channels={self.input_channels}, feature_channels={feature_channels}, border_attention={self.has_border_weights}")
        
        # Set Abstraction layers with 4x, 8x, 16x, 64x downsampling
        # SA1: 4x downsampling (2560 -> 640 points)
        self.sa1 = PointNetSetAbstractionMsg(640, [0.05, 0.1], [16, 32], feature_channels, [[32, 32, 64], [64, 64, 128]])
        
        # SA2: 8x downsampling (640 -> 320 points) 
        self.sa2 = PointNetSetAbstractionMsg(320, [0.1, 0.2], [16, 32], 192, [[64, 64, 128], [128, 128, 256]])
        
        # SA3: 16x downsampling (320 -> 160 points)
        self.sa3 = PointNetSetAbstractionMsg(160, [0.2, 0.4], [16, 32], 384, [[128, 128, 256], [256, 256, 512]])
        
        # SA4: 64x downsampling (160 -> 40 points)
        self.sa4 = PointNetSetAbstraction(40, 0.4, 32, 768 + 3, [256, 512, 1024], False)
        
        # Feature Propagation layers (upsampling)
        # FP4: 40 -> 160 points
        self.fp4 = PointNetFeaturePropagation(1024 + 768, [512, 512])
        
        # FP3: 160 -> 320 points  
        self.fp3 = PointNetFeaturePropagation(512 + 384, [512, 256])
        
        # FP2: 320 -> 640 points
        self.fp2 = PointNetFeaturePropagation(256 + 192, [256, 128])
        
        # FP1: 640 -> 2560 points (back to original)
        self.fp1 = PointNetFeaturePropagation(128 + feature_channels, [128, 128, 128])
        
        # Binary classification head for corner vs non-corner
        self.drop1 = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(128, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(64, 1, 1)  # Binary classification per point

    def forward(self, xyz):
        """
        Input:
            xyz: point cloud data [B, N, C] where C = 3 (XYZ) + features [+ border_weights]
        Output:
            corner_logits: [B, N] binary classification logits per point
            corner_features: [B, 128, N] per-point features
        """
        B, N, C = xyz.shape
        
        # Extract coordinates (first 3 dimensions)
        l0_xyz = xyz[:, :, 0:3].permute(0, 2, 1).contiguous()  # [B, 3, N]

        if self.has_border_weights and self.feature_channels > 0:
            # Split: features (dims 3 to C-1) and border weights (last dim)
            l0_features = xyz[:, :, 3:C-1].permute(0, 2, 1).contiguous()  # [B, feature_channels, N]
            self.border_weights = xyz[:, :, C-1:C].permute(0, 2, 1).contiguous()  # [B, 1, N]
            
            # Border attention will be applied AFTER feature learning in FP1
            l0_points = l0_features  # [B, feature_channels, N]
            
        elif self.feature_channels > 0:
            # Use all remaining channels as features
            l0_points = xyz[:, :, 3:C].permute(0, 2, 1).contiguous()  # [B, feature_channels, N]
        else:
            # No features at all (XYZ only)
            l0_points = None
        
        # Encoder: 4 Set Abstraction layers with downsampling
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        
        # Decoder: 4 Feature Propagation layers with upsampling
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)  
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        
        # Apply border attention to LEARNED features (after full encode-decode)
        if self.has_border_weights:
            # Constrained attention: weight bounded to [0, 0.12] via sigmoid
            constrained_weight = 0.12 * torch.sigmoid(self.border_attention_weight)
            
            # Compute attention boost
            border_boost = constrained_weight * self.border_weights  # [B, 1, N]
            border_boost = self.border_dropout(border_boost)  # Dropout on attention signal
            
            # Apply to LEARNED features: attended = features * (1 + border_boost)
            attention_factor = 1.0 + border_boost  # [B, 1, N]
            l0_points = l0_points * attention_factor  # [B, 128, N]
        
        # Binary classification head
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        corner_logits = self.conv2(feat).squeeze(1)  # [B, N]
        
        return corner_logits, l0_points