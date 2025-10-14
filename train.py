import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.PointNet2 import PointNet2CornerDetection
from losses import AdaptiveCornerLoss, create_corner_labels_improved
import os
import numpy as np

def train_model(train_loader, dataset_config):
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
    
    # NEW: Add geometric feature channels
    if getattr(dataset_config.Building3D, 'use_group_ids', False):
        input_channels += 1  # normalized group_id
        print("  Including group_ids as input channel")
    if getattr(dataset_config.Building3D, 'use_border_weights', False):
        input_channels += 1  # border_weights
        print("  Including border_weights as input channel")
    
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
    num_epochs = 300
    print(f"Starting training on {device}")
    print(f"Model input channels: {input_channels}")
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            point_clouds = batch['point_clouds'].to(device)  # [B, N, C]
            # Handle wf_vertices being either a padded tensor [B, M, 3] or a list of [Mi, 3]
            wf_vertices_batch = batch['wf_vertices']
            if isinstance(wf_vertices_batch, list):
                # Pad to the max number of corners with sentinel -1e0
                max_m = max(v.shape[0] for v in wf_vertices_batch) if len(wf_vertices_batch) > 0 else 0
                if max_m == 0:
                    wf_vertices = torch.full((point_clouds.shape[0], 1, 3), -1e0, device=device, dtype=point_clouds.dtype)
                else:
                    wf_vertices = torch.full((len(wf_vertices_batch), max_m, 3), -1e0, device=device, dtype=wf_vertices_batch[0].dtype)
                    for b, v in enumerate(wf_vertices_batch):
                        if v.numel() > 0:
                            m = v.shape[0]
                            wf_vertices[b, :m, :] = v.to(device)
            else:
                wf_vertices = wf_vertices_batch.to(device)    # [B, M, 3]
            
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
                'input_channels': input_channels,  # Save architecture info
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_model_path = 'output/corner_detection_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,  # Save architecture info
    }, final_model_path)
    print(f'Final model saved: {final_model_path}')
    
    return model

