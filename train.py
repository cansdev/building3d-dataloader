import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.PointNet2 import PointNet2CornerDetection
from losses import AdaptiveCornerLoss, create_corner_labels_improved
import os
import numpy as np
import yaml
from easydict import EasyDict
from datasets import build_dataset
from datasets.building3d import calculate_input_dim

def train_model(train_loader, dataset_config, input_channels=None):
    """
    Train PointNet2 model with preprocessed data
    
    Args:
        train_loader: DataLoader for training data
        dataset_config: Dataset configuration (full config object)
        input_channels: Pre-calculated input channels (if None, will calculate here for backward compatibility)
    """
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Access the Building3D nested config
    building3d_cfg = dataset_config.Building3D
    
    # Build model configuration dictionary
    model_config = {
        'use_color': building3d_cfg.use_color,
        'use_intensity': building3d_cfg.use_intensity,
        'use_group_ids': getattr(building3d_cfg, 'use_group_ids', False),
        'use_border_weights': getattr(building3d_cfg, 'use_border_weights', False),
        'normalize': building3d_cfg.normalize,
        'num_points': building3d_cfg.num_points,
        'input_channels': input_channels
    }
    
    print(f"Using {input_channels} input channels for PointNet2 model")
    model = PointNet2CornerDetection(config=model_config)
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
            
            # Include border attention weight if applicable
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'input_channels': input_channels,
                'config': model_config,
            }
            
            # Add learned border attention weight if enabled
            if model_config['use_border_weights'] and model.border_attention_weight is not None:
                raw_weight = model.border_attention_weight.item()
                constrained_weight = 0.2 * torch.sigmoid(torch.tensor(raw_weight)).item()
                checkpoint_data['border_attention_weight'] = raw_weight
                checkpoint_data['border_attention_weight_constrained'] = constrained_weight
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
            print(f'  Config: use_color={model_config["use_color"]}, '
                  f'use_intensity={model_config["use_intensity"]}, '
                  f'use_group_ids={model_config["use_group_ids"]}, '
                  f'use_border_weights={model_config["use_border_weights"]}')
            
            # Display learned border attention weight (both raw and constrained)
            if model_config['use_border_weights'] and model.border_attention_weight is not None:
                raw_weight = model.border_attention_weight.item()
                constrained_weight = 0.12 * torch.sigmoid(torch.tensor(raw_weight)).item()
                print(f'  Border attention: raw={raw_weight:.4f}, effective={constrained_weight:.4f} (max_boost=12%)')
    
    # Save final model
    final_model_path = 'output/corner_detection_model.pth'
    
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,
        'config': model_config,
    }
    
    # Add learned border attention weight if enabled
    if model_config['use_border_weights'] and model.border_attention_weight is not None:
        raw_weight = model.border_attention_weight.item()
        constrained_weight = 0.2 * torch.sigmoid(torch.tensor(raw_weight)).item()
        final_model_data['border_attention_weight'] = raw_weight
        final_model_data['border_attention_weight_constrained'] = constrained_weight
    
    torch.save(final_model_data, final_model_path)
    print(f'Final model saved: {final_model_path}')
    print(f'Model configuration:')
    print(f'  use_color: {model_config["use_color"]}')
    print(f'  use_intensity: {model_config["use_intensity"]}')
    print(f'  use_group_ids: {model_config["use_group_ids"]}')
    print(f'  use_border_weights: {model_config["use_border_weights"]}')
    print(f'  normalize: {model_config["normalize"]}')
    print(f'  num_points: {model_config["num_points"]}')
    print(f'  input_channels: {model_config["input_channels"]}')
    
    # Display learned border attention weight
    if model_config['use_border_weights'] and model.border_attention_weight is not None:
        raw_weight = model.border_attention_weight.item()
        constrained_weight = 0.12 * torch.sigmoid(torch.tensor(raw_weight)).item()
        print(f'  border_attention: raw={raw_weight:.4f}, effective={constrained_weight:.4f} (max_boost=12%)')
    print(f'  normalize: {model_config["normalize"]}')
    print(f'  num_points: {model_config["num_points"]}')
    print(f'  input_channels: {model_config["input_channels"]}')
    
    # Display learned border attention weight
    if model_config['use_border_weights'] and model.border_attention_weight is not None:
        print(f'  border_attention_weight: {model.border_attention_weight.item():.4f}')
    
    return model

def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)

    cfg = EasyDict(new_config)
    return cfg

def train_with_preprocessed_data():
    """
    Function that loads preprocessed data and passes it to train_model for PointNet2 training
    """
    # Load dataset configuration
    config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_config.yaml')
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Dynamically calculate and update input_dim using the function from building3d.py
    calculated_input_dim = calculate_input_dim(dataset_config.Building3D)
    
    print(f"Calculated input_dim: {calculated_input_dim}")
    print(f"  - XYZ: 3")
    print(f"  - Color (RGBA): {4 if dataset_config.Building3D.use_color else 0}")
    print(f"  - Intensity: {1 if dataset_config.Building3D.use_intensity else 0}")
    print(f"  - Group IDs: {1 if getattr(dataset_config.Building3D, 'use_group_ids', False) else 0}")
    print(f"  - Border weights: {1 if getattr(dataset_config.Building3D, 'use_border_weights', False) else 0}")
    
    # Build dataset with preprocessing
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Create data loaders for training and testing
    # Enable pin_memory for faster CPU->GPU transfer
    # Enable num_workers for parallel data loading
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=3, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch,
        pin_memory=True,      # Speeds up CPU->GPU transfer
        num_workers=4,        # Parallel data loading (adjust based on CPU cores)
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    # Call training function with preprocessed data using corner loss
    # Pass the calculated input_dim to the training function
    model = train_model(train_loader, dataset_config, calculated_input_dim)
    return model

def main():
    """
    Main program flow - calls training with preprocessed data
    """
    # Train the model
    train_with_preprocessed_data()

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()

