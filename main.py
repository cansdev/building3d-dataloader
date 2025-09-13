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
        batch_size=3, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch
    )
    
    test_loader = DataLoader(
        building3D_dataset['test'], 
        batch_size=3, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=building3D_dataset['test'].collate_batch
    )
    
    # Call training function with preprocessed data
    model = train_model(train_loader, test_loader, dataset_config)
    return model

def run_visualization():
    """
    Run PointNet2 visualization after training
    """
    print("\n" + "="*60)
    print("Training completed! Starting visualization...")
    print("="*60)
    
    try:
        # Run the visualization script
        visualize_script = os.path.join('visualize', 'visualize_pointnet2.py')
        if os.path.exists(visualize_script):
            print(f"Running {visualize_script}...")
            subprocess.run([sys.executable, visualize_script], check=True)
        else:
            print(f"Warning: Visualization script not found at {visualize_script}")
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
    except Exception as e:
        print(f"Unexpected error during visualization: {e}")

def main():
    """
    Main program flow - calls training with preprocessed data, then visualization
    """
    # Train the model
    model = train_with_preprocessed_data()
    
    # Ask user if they want to run visualization
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)
    
    # debugging visualization currently obsolete
    if False:
        while True:
            choice = input("\nDo you want to run PointNet2 visualization? (y/n): ").strip().lower()
            if choice in ['y', 'yes']:
                run_visualization()
                break
            elif choice in ['n', 'no']:
                print("Skipping visualization. You can run it later with:")
                print("python visualize/visualize_pointnet2.py")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")

if __name__ == "__main__":
    main()