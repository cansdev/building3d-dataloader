import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader

from datasets import build_dataset
import yaml
from easydict import EasyDict
from utils.cuda_utils import get_device, to_cuda, print_gpu_memory

# Import our training module
from train import train_on_real_dataset

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    return EasyDict(new_config)

def load_training_data(device=None, config_path=None):
    """Load and prepare all training data with full preprocessing"""

    # Load dataset with preprocessing enabled
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_config.yaml')
    dataset_config = cfg_from_yaml_file(config_path)
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    print(f"Dataset loaded: {len(building3D_dataset['train'])} samples")
    
    # Configure DataLoader with CUDA optimizations if using GPU
    use_cuda = device is not None and device.type == 'cuda'
    
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=4,
        shuffle=True,
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch,
        pin_memory=use_cuda
    )
    
    print(f"Total batches available: {len(train_loader)}")
    
    return train_loader

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

def main():
    """Main function to load training data and start training"""
    try:
        # Get the best available device (CUDA or CPU)
        device = get_device()
        print(f"Using device: {device}")
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_config.yaml')
        dataset_config = cfg_from_yaml_file(config_path)
        train_config = dataset_config.Training  # Get training config
        
        # Dynamically calculate and update input_dim
        calculated_input_dim = calculate_input_dim(dataset_config)
        train_config.model.input_dim = calculated_input_dim
        
        train_loader = load_training_data(device, config_path)

        print("\n" + "=" * 60)
        print("STARTING TRAINING SETUP")
        print("=" * 60)
        
        training_data = train_on_real_dataset(train_loader, device, train_config)

        print("\n" + "=" * 60)
        print("Training setup complete")
        print("=" * 60)

        return train_loader, training_data
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    train_loader, training_data = main()