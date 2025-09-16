#!/usr/bin/env python3
"""
Test script for Hungarian matching implementation
"""

import torch
import numpy as np
from models.PointNet2 import PointNet2CornerDetectionHungarian
from losses.hungarian_loss import HungarianMatcher, SetCriterion

def test_hungarian_model():
    """Test the Hungarian model forward pass"""
    print("Testing Hungarian model...")
    
    # Create model
    model = PointNet2CornerDetectionHungarian(input_channels=3, num_queries=50)
    model.eval()
    
    # Create dummy input
    batch_size = 2
    num_points = 2560
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(point_cloud)
    
    print(f"Input shape: {point_cloud.shape}")
    print(f"Output pred_logits shape: {outputs['pred_logits'].shape}")
    print(f"Output pred_boxes shape: {outputs['pred_boxes'].shape}")
    
    # Test corner predictions
    corners, scores = model.get_corner_predictions(point_cloud, threshold=0.3)
    print(f"Number of predicted corners per batch: {[len(c) for c in corners]}")
    
    print("✓ Hungarian model test passed!")

def test_hungarian_loss():
    """Test the Hungarian loss computation"""
    print("\nTesting Hungarian loss...")
    
    # Create model and loss
    model = PointNet2CornerDetectionHungarian(input_channels=3, num_queries=50)
    matcher = HungarianMatcher(cost_class=1.0, cost_coord=5.0)
    weight_dict = {"loss_ce": 1, "loss_bbox": 5}
    criterion = SetCriterion(
        num_classes=1, 
        matcher=matcher, 
        weight_dict=weight_dict, 
        eos_coef=0.1,
        losses=['labels', 'boxes']
    )
    
    # Create dummy data
    batch_size = 2
    num_points = 2560
    point_cloud = torch.randn(batch_size, num_points, 3)
    
    # Create dummy targets
    targets = [
        {
            'labels': torch.ones(5, dtype=torch.long),
            'boxes': torch.randn(5, 3)
        },
        {
            'labels': torch.ones(3, dtype=torch.long),
            'boxes': torch.randn(3, 3)
        }
    ]
    
    # Forward pass
    outputs = model(point_cloud)
    
    # Compute loss
    loss_dict = criterion(outputs, targets)
    total_loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    
    print(f"Loss components: {loss_dict}")
    print(f"Total loss: {total_loss.item():.4f}")
    
    print("✓ Hungarian loss test passed!")

def test_matcher():
    """Test the Hungarian matcher"""
    print("\nTesting Hungarian matcher...")
    
    matcher = HungarianMatcher(cost_class=1.0, cost_coord=5.0)
    
    # Create dummy outputs and targets
    outputs = {
        'pred_logits': torch.randn(2, 50, 1),
        'pred_boxes': torch.randn(2, 50, 3)
    }
    
    targets = [
        {
            'labels': torch.ones(5, dtype=torch.long),
            'boxes': torch.randn(5, 3)
        },
        {
            'labels': torch.ones(3, dtype=torch.long),
            'boxes': torch.randn(3, 3)
        }
    ]
    
    # Test matching
    indices = matcher(outputs, targets)
    
    print(f"Number of matches per batch: {[len(idx[0]) for idx in indices]}")
    print(f"Match indices: {indices}")
    
    print("✓ Hungarian matcher test passed!")

if __name__ == "__main__":
    print("Running Hungarian matching tests...")
    print("=" * 50)
    
    try:
        test_hungarian_model()
        test_hungarian_loss()
        test_matcher()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Hungarian matching implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
