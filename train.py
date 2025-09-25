
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.PointNet2 import PointNet2CornerDetection
from losses import AdaptiveCornerLoss, create_corner_labels
import os
import numpy as np


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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    
    # Use adaptive corner detection loss
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
            corner_labels = create_corner_labels(
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
        
        # Handle case where no batches were processed
        if num_batches == 0:
            print(f'Epoch {epoch} completed. No batches processed - check data loading!')
            continue
            
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
