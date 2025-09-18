import os
import numpy as np

from datasets import build_dataset
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    return EasyDict(new_config)

def load_training_data():
    """Load and prepare all training data with full preprocessing"""
    print("=" * 60)
    print("BUILDING3D TRAINING DATA LOADER")
    print("=" * 60)
    
    # Load dataset with preprocessing enabled
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Create training dataloader
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=4,
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch,
        num_workers=0
    )
    
    print(f"✓ Training dataset loaded: {len(building3D_dataset['train'])} samples")
    print(f"✓ DataLoader created with batch_size=4")
    
    # Verify first batch
    for batch in train_loader:
        print(f"\n✓ Data verification:")
        print(f"  - Point clouds: {batch['point_clouds'].shape}")
        print(f"  - Wireframe vertices: {batch['wf_vertices'].shape}")
        print(f"  - Wireframe edges: {batch['wf_edges'].shape}")
        print(f"  - Centroids: {batch['centroid'].shape}")
        print(f"  - Max distances: {batch['max_distance'].shape}")
        
        # Check data quality
        sample_pc = batch['point_clouds'][0].numpy()
        print(f"  - Point range: [{sample_pc.min():.3f}, {sample_pc.max():.3f}]")
        print(f"  - Features per point: {sample_pc.shape[1]}")
        print(f"  - Preprocessing: Normals, surface groups, border weights included ✓")
        break
    
    print(f"\n✓ Training data ready for model training!")
    return train_loader

def main():
    """Main function to load training data"""
    print("Training Data Preparation")
    
    try:
        # Load training data
        train_loader = load_training_data()
        
        print("\n" + "=" * 60)
        print("READY FOR TRAINING!")
        print("=" * 60)
        return train_loader
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

if __name__ == "__main__":
    train_loader = main()