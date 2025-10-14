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

def train_with_preprocessed_data():
    """
    Function that loads preprocessed data and passes it to train.py for PointNet2 training
    """
    # Load dataset configuration
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    
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
    model = train_model(train_loader, dataset_config)
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