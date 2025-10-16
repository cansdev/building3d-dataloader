import os
import subprocess
import sys

from datasets import build_dataset
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from train import train_model

def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)

        cfg = EasyDict(new_config)
        return cfg

def calculate_input_dim(dataset_config):
    """Dynamically calculate input dimensions based on enabled features"""
    input_dim = 3  # XYZ coordinates (always present)
    
    # Add color channels
    if dataset_config.Building3D.use_color:
        input_dim += 4  # RGBA
    
    # Add intensity channel
    if dataset_config.Building3D.use_intensity:
        input_dim += 1  # Intensity
    
    # Add geometric features if enabled
    if getattr(dataset_config.Building3D, 'use_group_ids', False):
        input_dim += 1  # Normalized group IDs
    
    if getattr(dataset_config.Building3D, 'use_border_weights', False):
        input_dim += 1  # Border weights
    
    return input_dim

def train_with_preprocessed_data():
    """
    Function that loads preprocessed data and passes it to train.py for PointNet2 training
    """
    # Load dataset configuration
    config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_config.yaml')
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Dynamically calculate and update input_dim
    calculated_input_dim = calculate_input_dim(dataset_config)
    
    print(f"Calculated input_dim: {calculated_input_dim}")
    print(f"  - XYZ: 3")
    print(f"  - Color (RGBA): {4 if dataset_config.Building3D.use_color else 0}")
    print(f"  - Intensity: {1 if dataset_config.Building3D.use_intensity else 0}")
    print(f"  - Group IDs: {1 if getattr(dataset_config.Building3D, 'use_group_ids', False) else 0}")
    print(f"  - Border weights: {1 if getattr(dataset_config.Building3D, 'use_border_weights', False) else 0}")
    
    # Build dataset with preprocessing
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Create data loaders for training and testing
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=12, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch
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