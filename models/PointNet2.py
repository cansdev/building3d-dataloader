import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet2_utils import PointNetSetAbstraction, PointNetSetAbstractionMsg, PointNetFeaturePropagation


class PointNet2(nn.Module):
    def __init__(self, num_class=40, normal_channel=True):
        super(PointNet2, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        
        # Set Abstraction layers (encoder)
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
        # Feature Propagation layers (decoder)
        self.fp3 = PointNetFeaturePropagation(1664, [256, 256])
        self.fp2 = PointNetFeaturePropagation(576, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channel, [128, 128, 128])
        
        # Classification head
        self.drop1 = nn.Dropout(0.5)
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz):
        """
        Input:
            xyz: point cloud data [B, N, C] where C is coordinate + features
        Output:
            features: per-point features [B, num_class, N]
        """
        B, N, C = xyz.shape
        if self.normal_channel:
            norm = xyz[:, :, 3:6]
        else:
            norm = None
        
        # Extract coordinates
        xyz_only = xyz[:, :, 0:3]
        
        # Convert to [B, C, N] format for PointNet++
        xyz_only = xyz_only.permute(0, 2, 1).contiguous()
        if norm is not None:
            norm = norm.permute(0, 2, 1).contiguous()
        
        # Encoder (Set Abstraction)
        l0_points = norm
        l0_xyz = xyz_only
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        # Decoder (Feature Propagation)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, norm], 1) if norm is not None else l0_xyz, l1_points)
        
        # Classification head
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        feat = self.conv2(feat)
        
        return feat

    def get_loss(self, pred, target, trans_feat=None):
        """
        Simple cross-entropy loss for demonstration
        """
        total_loss = F.cross_entropy(pred, target)
        return total_loss


class PointNet2CornerDetection(nn.Module):
    """
    PointNet2 for corner point detection with specific SA/FP architecture
    """
    def __init__(self, input_channels=3):
        super(PointNet2CornerDetection, self).__init__()
        # Feature channels = total channels - 3 (xyz coordinates)
        feature_channels = max(0, input_channels - 3)
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        
        print(f"    CornerNet init: total_channels={input_channels}, feature_channels={feature_channels}")
        
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
            xyz: point cloud data [B, N, C] 
        Output:
            corner_logits: per-point corner predictions [B, N]
            corner_features: per-point features [B, 128, N]
        """
        B, N, C = xyz.shape
        
        # Extract coordinates (first 3 dimensions)
        l0_xyz = xyz[:, :, 0:3].permute(0, 2, 1).contiguous()
        
        # Use additional features if available (C > 3)
        if C > 3:
            l0_points = xyz[:, :, 3:C].permute(0, 2, 1).contiguous()
        else:
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
        
        # Point-wise classification head
        feat = F.relu(self.bn1(self.conv1(l0_points)))
        feat = self.drop1(feat)
        corner_logits = self.conv2(feat)  # [B, 1, N]
        corner_logits = corner_logits.squeeze(1)  # [B, N]
        
        return corner_logits, l0_points
    
    def get_corner_predictions(self, xyz, threshold=0.5):
        """
        Get predicted corner points above threshold
        
        Returns:
            corner_points: [B, K, 3] where K is number of predicted corners
            corner_scores: [B, K] confidence scores
        """
        with torch.no_grad():
            corner_logits, _ = self.forward(xyz)
            corner_probs = torch.sigmoid(corner_logits)  # [B, N]
            
            batch_corners = []
            batch_scores = []
            
            for b in range(xyz.shape[0]):
                # Get points above threshold
                mask = corner_probs[b] > threshold
                if mask.sum() > 0:
                    corner_coords = xyz[b, mask, :3]  # [K, 3]
                    corner_confidences = corner_probs[b, mask]  # [K]
                    batch_corners.append(corner_coords)
                    batch_scores.append(corner_confidences)
                else:
                    # No corners found
                    batch_corners.append(torch.zeros(0, 3, device=xyz.device))
                    batch_scores.append(torch.zeros(0, device=xyz.device))
            
            return batch_corners, batch_scores
