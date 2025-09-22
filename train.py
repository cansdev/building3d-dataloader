#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# Training with DGCNN model and CUDA support

import torch
import torch.nn as nn
import numpy as np
from utils.cuda_utils import to_cuda, print_gpu_memory
from models.dgcnn_model import BasicDGCNN

def main_training_setup():
    """Setup training configuration"""
    print("Training setup initialized with DGCNN")
    return True

def train_on_real_dataset(train_loader, device):
    """
    Train DGCNN model on vertex coordinate prediction with CUDA support.
    
    Args:
        train_loader: DataLoader for training data
        device: Device to use (cuda or cpu)
    
    Returns:
        dict: Training results
    """
    print(f"Training DGCNN for vertex coordinate prediction with {device}")
    
    # Initialize DGCNN model for 5 features: X Y Z GroupID BorderWeight
    model = BasicDGCNN(input_dim=5, k=20, max_vertices=64)
    model = model.to(device)
    
    # Optimizer for coordinate regression - Start with higher LR, will reduce on plateau
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)  # Higher initial LR with light regularization
    
    # Learning rate scheduler - Reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True, min_lr=1e-5
    )
    
    # Improved loss functions
    coord_loss_fn = nn.SmoothL1Loss()  # More robust than MSE, less sensitive to outliers
    num_vertices_loss_fn = nn.MSELoss()  # Keep MSE for count prediction
    
    print(f"DGCNN model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Find the first sample (1.xyz and 1.obj)
    single_batch = None
    for batch_idx, batch_data in enumerate(train_loader):
        batch_on_device = to_cuda(batch_data, device)
        
        # Check if this is sample "1" (scan_idx == 1)
        if 'scan_idx' in batch_on_device:
            scan_indices = batch_on_device['scan_idx']
            if 1 in scan_indices:
                # Find the position of sample 1 in the batch
                sample_pos = (scan_indices == 1).nonzero(as_tuple=True)[0][0]
                
                # Extract just that one sample
                single_batch = {}
                for key, value in batch_on_device.items():
                    if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                        single_batch[key] = value[sample_pos:sample_pos+1]  # Keep batch dimension
                    else:
                        single_batch[key] = value
                
                print(f"Found sample 1.xyz in batch {batch_idx + 1}")
                break
    
    if single_batch is None:
        print("Sample 1.xyz not found in dataset")
        return {'error': 'Sample 1 not found'}
    
    # Extract data
    points = single_batch['point_clouds']  # [1, N, 5]
    vertices_gt = single_batch['wf_vertices']  # [1, V, 3] ground truth vertex coordinates
    
    print(f"Training on sample 1:")
    print(f"   Point cloud: {points.shape}")
    print(f"   GT vertices: {vertices_gt.shape}")
    
    # Prepare ground truth
    B, V, _ = vertices_gt.shape
    num_vertices_gt = torch.tensor([V], dtype=torch.float32, device=device)  # [1]
    
    # Pad vertices to max_vertices if needed
    max_vertices = model.max_vertices
    if V > max_vertices:
        print(f"Warning: GT has {V} vertices, but model max is {max_vertices}. Truncating.")
        vertices_gt = vertices_gt[:, :max_vertices, :]
        num_vertices_gt = torch.tensor([max_vertices], dtype=torch.float32, device=device)
        V = max_vertices
    
    # Create padded ground truth tensor
    vertices_gt_padded = torch.zeros(B, max_vertices, 3, device=device)
    vertices_gt_padded[:, :V, :] = vertices_gt[:, :V, :]
    
    # Extract feature information
    points_xyz = points[:, :, :3]      # [1, N, 3] - XYZ coordinates
    group_ids = points[:, :, 3:4]      # [1, N, 1] - Group IDs
    border_weights = points[:, :, 4:5] # [1, N, 1] - Border weights
    
    print(f"   Feature analysis:")
    print(f"   Point cloud shape: {points.shape}")
    print(f"   Group IDs range: {group_ids.min():.1f} to {group_ids.max():.1f}")
    print(f"   Border weights range: {border_weights.min():.3f} to {border_weights.max():.3f}")
    print(f"   Ground truth vertices: {V}")
    
    # Training loop for OVERFITTING
    model.train()
    num_epochs = 500  # More epochs for overfitting
    
    print(f"Starting OVERFITTING training for {num_epochs} epochs...")
    print(f"Goal: Perfectly overfit on sample 1 to test model capacity")
    
    best_loss = float('inf')
    best_vertex_error = float('inf')
    
    # Adaptive loss weighting - start with coordinate focus, gradually balance
    base_coord_weight = 1.0
    base_num_weight = 0.1
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(points)
        pred_vertices = outputs['vertex_coords']  # [1, max_vertices, 3]
        pred_num_vertices_sigmoid = outputs['num_vertices_sigmoid']  # [1] - sigmoid output
        
        # Coordinate loss (only for valid vertices)
        coord_loss = coord_loss_fn(pred_vertices[:, :V, :], vertices_gt_padded[:, :V, :])
        
        # Number of vertices loss - NOW USING SIGMOID TARGET
        num_vertices_target = num_vertices_gt / model.max_vertices  # Convert to 0-1 range
        num_loss = num_vertices_loss_fn(pred_num_vertices_sigmoid, num_vertices_target)
        
        # Adaptive loss weighting - gradually increase number prediction importance
        progress = epoch / num_epochs
        coord_weight = base_coord_weight
        num_weight = base_num_weight + (0.5 * progress)  # Gradually increase from 0.1 to 0.6
        
        # Combined loss with adaptive weighting
        total_loss = coord_weight * coord_loss + num_weight * num_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Step learning rate scheduler
        scheduler.step(total_loss)
        
        # Track best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
        
        # Print progress more frequently for overfitting
        if (epoch + 1) % 100 == 0 or epoch < 10:
            with torch.no_grad():
                pred_num_sigmoid = outputs['num_vertices_sigmoid'][0].item()  # This is already sigmoid output
                pred_num = pred_num_sigmoid * model.max_vertices  # Scale to vertex count
                
                # Compute vertex distance error
                vertex_dist = torch.norm(pred_vertices[:, :V, :] - vertices_gt_padded[:, :V, :], dim=2)
                mean_vertex_error = vertex_dist.mean().item()
                max_vertex_error = vertex_dist.max().item()
                
                # Count accurate vertices
                accurate_vertices = torch.sum(vertex_dist < 0.05).item()  # Within 5cm
                very_accurate = torch.sum(vertex_dist < 0.01).item()      # Within 1cm
                
                if mean_vertex_error < best_vertex_error:
                    best_vertex_error = mean_vertex_error
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                print(f"Epoch {epoch+1:4d}: Loss={total_loss.item():.6f} "
                      f"(Coord={coord_loss.item():.6f}×{coord_weight:.1f}, Num={num_loss.item():.6f}×{num_weight:.1f}), "
                      f"LR={current_lr:.2e}, Pred_num={pred_num:.1f}, Mean_err={mean_vertex_error:.4f}, "
                      f"Acc_verts={accurate_vertices}/{V}, Very_acc={very_accurate}/{V}")
                
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(points)
        pred_vertices = outputs['vertex_coords']  # [1, max_vertices, 3]
        pred_num_vertices = outputs['num_vertices'][0].item()
        
        # Compute final metrics
        final_coord_loss = coord_loss_fn(pred_vertices[:, :V, :], vertices_gt_padded[:, :V, :])
        vertex_distances = torch.norm(pred_vertices[:, :V, :] - vertices_gt_padded[:, :V, :], dim=2)
        mean_vertex_error = vertex_distances.mean().item()
        max_vertex_error = vertex_distances.max().item()
        
        # Count accurate vertices for final analysis
        accurate_vertices = torch.sum(vertex_distances < 0.05).item()  # Within 5cm
        very_accurate = torch.sum(vertex_distances < 0.01).item()      # Within 1cm
        
        print(f"Training complete!")
        print(f"   Best vertex error achieved: {best_vertex_error:.6f}")
        print(f"   Final coordinate loss: {final_coord_loss.item():.6f}")
        print(f"   Mean vertex error: {mean_vertex_error:.4f}")
        print(f"   Max vertex error: {max_vertex_error:.4f}")
        print(f"   Predicted vertices: {pred_num_vertices:.1f}")
        print(f"   Ground truth vertices: {V}")
        print(f"   Vertices within 5cm: {accurate_vertices}/{V}")
        print(f"   Vertices within 1cm: {very_accurate}/{V}")
        
        # Detailed analysis of overfitting success
        if mean_vertex_error < 0.01:
            print(f"EXCELLENT overfitting: Mean error < 1cm")
        elif mean_vertex_error < 0.05:
            print(f"GOOD overfitting: Mean error < 5cm")
        elif mean_vertex_error < 0.1:
            print(f"MODERATE overfitting: Mean error < 10cm")
        else:
            print(f"POOR overfitting: Mean error > 10cm - architecture issues?")
        
        if abs(pred_num_vertices - V) < 1.0:
            print(f"Number prediction: EXCELLENT (error: {abs(pred_num_vertices - V):.1f})")
        else:
            print(f"Number prediction: FAILED (error: {abs(pred_num_vertices - V):.1f})")
        
        # Show some example predictions vs ground truth
        print(f"Example vertex predictions (first 5):")
        for i in range(min(5, V)):
            gt = vertices_gt_padded[0, i, :].cpu().numpy()
            pred = pred_vertices[0, i, :].cpu().numpy()
            error = vertex_distances[0, i].item()
            print(f"   Vertex {i}: GT=[{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}] "
                  f"Pred=[{pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f}] Error={error:.4f}")
    
    # Show GPU memory
    if device.type == 'cuda':
        print_gpu_memory()
    
    return {
        'model': model,
        'final_coord_loss': final_coord_loss.item(),
        'mean_vertex_error': mean_vertex_error,
        'max_vertex_error': max_vertex_error,
        'predicted_num_vertices': pred_num_vertices,
        'ground_truth_vertices': V,
        'device_used': str(device),
        'sample_trained': '1.xyz/1.obj'
    }