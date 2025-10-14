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

def load_training_data(device=None):
    """Load and prepare all training data with full preprocessing"""

    # Load dataset with preprocessing enabled
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    print(f"Dataset loaded: {len(building3D_dataset['train'])} samples")
    
    # Configure DataLoader with CUDA optimizations if using GPU
    use_cuda = device is not None and device.type == 'cuda'
    
    # Create deterministic generator for DataLoader
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=4,
        shuffle=False,
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch,
        pin_memory=use_cuda,  # Pin memory for faster GPU transfer
        num_workers=0,  # Keep simple for now
        generator=generator  # Deterministic sampling
    )
    
    print(f"Total batches available: {len(train_loader)}")
    
    return train_loader

def main():
    """Main function to load training data and start training"""
    try:
        # Get the best available device (CUDA or CPU)
        device = get_device()
        print(f"Using device: {device}")
        
        # Load configuration
        dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
        train_config = dataset_config.Training  # Get training config
        
        train_loader = load_training_data(device)

        print("\n" + "=" * 60)
        print("STARTING TRAINING SETUP")
        print("=" * 60)
        
        training_data = train_on_real_dataset(train_loader, device, train_config)

        # Save the trained model
        if 'model' in training_data:
            model_save_path = 'trained_dgcnn_model.pth'
            torch.save(training_data['model'].state_dict(), model_save_path)
            print(f"\nTrained model saved to: {model_save_path}")

        print("\n" + "=" * 60)
        print("Training setup complete")
        print("=" * 60)

        return train_loader, training_data
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

if __name__ == "__main__":
    train_loader, training_data = main()