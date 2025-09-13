import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.PointNet2 import PointNet2CornerDetection
from losses import CornerDetectionLoss, AdaptiveCornerLoss, create_corner_labels_improved
import os
import numpy as np

def create_corner_labels(point_clouds, wf_vertices, distance_threshold=0.05):
    """
    Create corner labels by finding point cloud points closest to wireframe vertices
    
    Args:
        point_clouds: [B, N, C] - point cloud data
        wf_vertices: [B, M, 3] - wireframe vertices (corners)
        distance_threshold: maximum distance to consider a point as a corner
    
    Returns:
        corner_labels: [B, N] - binary labels (1 for corner, 0 for non-corner)
    """
    B, N, C = point_clouds.shape
    corner_labels = torch.zeros(B, N, device=point_clouds.device)
    
    # Extract XYZ coordinates from point clouds
    pc_xyz = point_clouds[:, :, :3]  # [B, N, 3]
    
    for b in range(B):
        # Get valid wireframe vertices (not padded with -1e1)
        valid_mask = wf_vertices[b, :, 0] > -1e0  # Check if not padded
        valid_wf_vertices = wf_vertices[b, valid_mask]  # [M_valid, 3]
        
        if len(valid_wf_vertices) == 0:
            continue  # No valid wireframe vertices for this batch
            
        # Compute distances from each point cloud point to all wireframe vertices
        pc_points = pc_xyz[b]  # [N, 3]
        wf_points = valid_wf_vertices  # [M_valid, 3]
        
        # Compute pairwise distances: [N, M_valid]
        distances = torch.cdist(pc_points, wf_points, p=2)
        
        # Find minimum distance for each point cloud point
        min_distances, _ = torch.min(distances, dim=1)  # [N]
        
        # Mark points as corners if they're within threshold distance
        corner_mask = min_distances < distance_threshold
        corner_labels[b, corner_mask] = 1.0
    
    return corner_labels

def train_model(train_loader, test_loader, dataset_config):
    """
    Train PointNet2 model with preprocessed data
    
    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data  
        dataset_config: Dataset configuration
    """
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determine input channels based on dataset config
    input_channels = 3  # xyz coordinates
    if dataset_config.Building3D.use_color:
        input_channels += 4  # rgba
    if dataset_config.Building3D.use_intensity:
        input_channels += 1  # intensity
    
    model = PointNet2CornerDetection(input_channels=input_channels)
    model = model.to(device)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Use comprehensive corner detection loss
    criterion = AdaptiveCornerLoss(
        initial_focal_gamma=2.0,
        min_focal_gamma=0.5
    )
    
    # Training parameters
    num_epochs = 1000
    print(f"Starting training on {device}")
    print(f"Model input channels: {input_channels}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            point_clouds = batch['point_clouds'].to(device)  # [B, N, C]
            wf_vertices = batch['wf_vertices'].to(device)    # [B, M, 3]
            
            # Create improved corner labels with soft labels
            corner_labels = create_corner_labels_improved(
                point_clouds, wf_vertices, 
                distance_threshold=0.05, 
                soft_labels=True
            )
            
            # Log corner statistics
            num_corners = (corner_labels > 0.5).sum().item()
            total_points = corner_labels.numel()
            corner_ratio = num_corners / total_points if total_points > 0 else 0
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            corner_logits, _ = model(point_clouds)  # [B, N]
            
            # Compute comprehensive loss
            loss_dict = criterion(corner_logits, corner_labels, point_clouds, wf_vertices)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.4f}, '
                      f'Focal: {loss_dict["focal_loss"].item():.4f}, '
                      f'Distance: {loss_dict["distance_loss"].item():.4f}, '
                      f'Corners: {num_corners:.0f}/{total_points} ({corner_ratio:.3f})')
        
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}')
        
        # Update adaptive loss for next epoch
        criterion.update_epoch(epoch)
        
        # Save model checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'output/checkpoint_epoch_{epoch}.pth'
            os.makedirs('output', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_model_path = 'output/corner_detection_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'Final model saved: {final_model_path}')
    
    return model
