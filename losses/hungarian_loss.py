import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """
    Hungarian matching for optimal assignment between predicted and ground truth corners
    """
    def __init__(self, cost_class=1.0, cost_coord=5.0):
        super(HungarianMatcher, self).__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, 1] and 'pred_boxes' [B, N, 3]
            targets: list of dicts with 'labels' [M] and 'boxes' [M, 3]
        Returns:
            List of (pred_idx, target_idx) pairs for each batch element
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        
        # Flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*N, 1]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*N, 3]
        
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        
        # Compute the classification cost
        # For corner detection, we use negative log probability as cost
        # Higher probability = lower cost
        cost_class = -torch.log(out_prob + 1e-8)  # [B*N, 1]
        cost_class = cost_class.squeeze(-1)  # [B*N] - remove the last dimension
        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Final cost matrix
        # Expand cost_class to match cost_bbox dimensions
        cost_class_expanded = cost_class.unsqueeze(1).expand(-1, tgt_bbox.shape[0])  # [B*N, M]
        C = self.cost_coord * cost_bbox + self.cost_class * cost_class_expanded
        C = C.view(bs, num_queries, -1).cpu()
        
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) 
                for i, j in indices]


class HungarianCornerLoss(nn.Module):
    """
    Hungarian loss for corner detection with optimal assignment
    """
    def __init__(self, num_classes=1, matcher=None, weight_dict=None, eos_coef=0.1):
        super(HungarianCornerLoss, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.weight_dict = weight_dict if weight_dict is not None else {
            "loss_ce": 1, "loss_bbox": 5
        }
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef  # Background class weight
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, 1] and 'pred_boxes' [B, N, 3]
            targets: list of dicts with 'labels' [M] and 'boxes' [M, 3]
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        
        # Compute the average number of target boxes across all nodes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        losses.update(self.get_loss(outputs, targets, indices, num_boxes))
        
        return losses

    def get_loss(self, outputs, targets, indices, num_boxes):
        """
        Compute losses for matched predictions
        """
        losses = {}
        
        # Classification loss
        src_logits = outputs['pred_logits'].squeeze(-1)  # [B, N] - remove last dimension
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape, dtype=torch.float, device=src_logits.device)
        target_classes[idx] = target_classes_o.float()
        
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses['loss_ce'] = loss_ce
        
        # Bbox regression loss
        src_boxes = outputs['pred_boxes']  # [B, N, 3]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes[idx], target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        return losses

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class SetCriterion(nn.Module):
    """
    Set-based criterion for corner detection
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super(SetCriterion, self).__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[0] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict with 'pred_logits' [B, N, 1] and 'pred_boxes' [B, N, 3]
            targets: list of dicts with 'labels' [M] and 'boxes' [M, 3]
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        
        # Compute the average number of target boxes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (Binary Cross Entropy)"""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].squeeze(-1)  # [B, N] - remove last dimension
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros(src_logits.shape, dtype=torch.float, device=src_logits.device)
        target_classes[idx] = target_classes_o.float()
        
        # Use binary cross entropy loss
        loss_ce = F.binary_cross_entropy_with_logits(src_logits, target_classes)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes"""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
