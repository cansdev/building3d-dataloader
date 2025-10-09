#!/usr/bin/python3
# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class VertexLossHungarian(nn.Module):
    """
    Combined loss: Hungarian matching + MSE for coordinates, BCE for existence.
    DETR-style optimal bipartite matching for handling vertex correspondence.
    """
    
    def __init__(self, coord_weight=10.0, existence_weight=1.0):
        super(VertexLossHungarian, self).__init__()
        
        self.coord_weight = coord_weight
        self.existence_weight = existence_weight
        
        # Loss functions
        self.existence_loss_fn = nn.BCEWithLogitsLoss()
    
    def hungarian_matching_loss(self, pred, gt):
        """
        DETR-style: L1 cost matrix → Hungarian matching → MSE loss.
        """
        # L1 cost matrix (more stable than L2)
        cost_matrix = torch.cdist(pred, gt, p=1)  # [N, N]
        
        # Optimal assignment (scipy, no gradient)
        with torch.no_grad():
            cost_np = cost_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_np)
            col_ind = torch.tensor(col_ind, dtype=torch.long, device=pred.device)
        
        # Reorder GT and compute MSE
        gt_matched = gt[col_ind]
        loss = F.mse_loss(pred, gt_matched)
        
        return loss
    
    def forward(self, predictions, targets, num_vertices_list):
        """Compute combined loss: existence BCE + Hungarian-matched coordinate MSE."""
        device = predictions['vertex_offsets'].device
        B = predictions['vertex_offsets'].shape[0]
        
        # Extract predictions and targets
        pred_offsets = predictions['vertex_offsets']
        existence_logits = predictions['existence_logits']
        vertices_gt_offsets = targets['vertices_gt_offsets']
        existence_target = targets['existence_target']
        
        # 1. Existence loss (BCE)
        existence_loss = self.existence_loss_fn(existence_logits, existence_target)
        
        # 2. Coordinate loss (Hungarian matching per sample)
        coord_losses = []
        for i in range(B):
            V_i = num_vertices_list[i]
            if isinstance(V_i, torch.Tensor):
                V_i = V_i.item()
            
            if V_i > 0:
                pred_vertices_i = pred_offsets[i, :V_i, :]
                gt_vertices_i = vertices_gt_offsets[i, :V_i, :]
                coord_loss_i = self.hungarian_matching_loss(pred_vertices_i, gt_vertices_i)
                coord_losses.append(coord_loss_i)
        
        if len(coord_losses) > 0:
            coord_loss = torch.stack(coord_losses).mean()
        else:
            coord_loss = torch.tensor(0.0, device=device)
        
        # 3. Combined loss with weighting
        coord_loss_weighted = self.coord_weight * coord_loss
        existence_loss_weighted = self.existence_weight * existence_loss
        total_loss = coord_loss_weighted + existence_loss_weighted
        
        return {
            'total_loss': total_loss,
            'coord_loss': coord_loss,
            'existence_loss': existence_loss,
            'coord_loss_weighted': coord_loss_weighted,
            'existence_loss_weighted': existence_loss_weighted
        }
    
    def update_weights(self, coord_weight=None, existence_weight=None):
        """Update loss weights dynamically during training."""
        if coord_weight is not None:
            self.coord_weight = coord_weight
        if existence_weight is not None:
            self.existence_weight = existence_weight
