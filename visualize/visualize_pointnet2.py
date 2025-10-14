#!/usr/bin/env python3
"""
PointNet2 Corner Detection Visualization Script

This script loads a trained PointNet2 model and visualizes its corner detection results
on the Building3D dataset. It shows:
1. Original point cloud
2. Ground truth wireframe vertices (corners)
3. Predicted corners from PointNet2
4. Comparison between ground truth and predictions
"""

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
CORNER_THRESHOLD = 0.8 # Threshold for corner predictions (0.0-1.0)
# =============================================================================

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset
from models.PointNet2 import PointNet2CornerDetection
import yaml
from easydict import EasyDict

def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg

def load_trained_model(model_path, input_channels, device):
    """Load trained PointNet2 model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint formats: full checkpoint dict or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: checkpoint contains model_state_dict and metadata
        saved_input_channels = checkpoint.get('input_channels', None)
        
        if saved_input_channels is not None and saved_input_channels != input_channels:
            print(f"⚠️  WARNING: Model was trained with {saved_input_channels} input channels, "
                  f"but config specifies {input_channels} channels!")
            print(f"    Using the model's trained architecture ({saved_input_channels} channels).")
            input_channels = saved_input_channels
        
        model = PointNet2CornerDetection(input_channels=input_channels)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Model architecture: {input_channels} input channels")
    else:
        # Old format: checkpoint is just the state_dict
        print("⚠️  WARNING: Old checkpoint format detected. Cannot verify input channels.")
        print(f"    Assuming {input_channels} input channels from config.")
        model = PointNet2CornerDetection(input_channels=input_channels)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def visualize_pointnet2_results(model, dataset, sample_idx, device, threshold=CORNER_THRESHOLD):
    """
    Visualize PointNet2 corner detection results for a specific sample
    
    Args:
        model: Trained PointNet2 model
        dataset: Building3D dataset
        sample_idx: Index of sample to visualize
        device: PyTorch device
        threshold: Threshold for corner predictions
    """
    # Get sample data
    sample = dataset[sample_idx]
    
    # Convert numpy arrays to tensors if needed
    if isinstance(sample['point_clouds'], np.ndarray):
        point_clouds = torch.from_numpy(sample['point_clouds']).unsqueeze(0).to(device)  # [1, N, C]
    else:
        point_clouds = sample['point_clouds'].unsqueeze(0).to(device)  # [1, N, C]
    
    if isinstance(sample['wf_vertices'], np.ndarray):
        wf_vertices = torch.from_numpy(sample['wf_vertices']).unsqueeze(0).to(device)    # [1, M, 3]
    else:
        wf_vertices = sample['wf_vertices'].unsqueeze(0).to(device)    # [1, M, 3]
    
    # Get ground truth corners (wireframe vertices)
    valid_mask = wf_vertices[0, :, 0] > -1e0
    gt_corners = wf_vertices[0, valid_mask].cpu().numpy()  # [M_valid, 3]
    
    # Get PointNet2 predictions
    with torch.no_grad():
        outputs = model(point_clouds)
        # Handle Hungarian model (dict output with query-based corners)
        if isinstance(outputs, dict) and 'pred_logits' in outputs and 'pred_boxes' in outputs:
            # Query-based predictions: get corner coords and confidences
            batch_corners, batch_scores = model.get_corner_predictions(point_clouds, threshold=threshold)
            predicted_corner_coords = batch_corners[0].cpu().numpy()  # [K, 3]
            predicted_corner_scores = batch_scores[0].cpu().numpy()   # [K]
            corner_probs = None
            corner_predictions = None
        else:
            # Per-point logits model
            corner_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            corner_probs = torch.sigmoid(corner_logits[0])  # [N]
            corner_predictions = (corner_probs > threshold).cpu().numpy()  # [N]
    
    # Get point cloud coordinates
    pc_xyz = point_clouds[0, :, :3].cpu().numpy()  # [N, 3]
    if corner_predictions is not None:
        predicted_corners = pc_xyz[corner_predictions]  # [K, 3]
    else:
        predicted_corners = predicted_corner_coords  # [K, 3]
    
    # Create visualization
    fig = plt.figure(figsize=(18, 6))
    
    # Plot 1: Original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                c='lightblue', s=1, alpha=0.6, label='Point Cloud')
    ax1.set_title('Original Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    
    # Plot 2: Ground truth corners
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                c='lightblue', s=1, alpha=0.3, label='Point Cloud')
    if len(gt_corners) > 0:
        ax2.scatter(gt_corners[:, 0], gt_corners[:, 1], gt_corners[:, 2], 
                    c='red', s=50, marker='o', label='GT Corners')
    ax2.set_title('Ground Truth Corners')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Plot 3: Predicted corners with probability coloring (if available)
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Color points by corner probability if per-point probs exist; otherwise plain
    if corner_probs is not None:
        scatter = ax3.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                             c=corner_probs.cpu().numpy(), cmap='viridis', s=2, alpha=0.8)
    else:
        scatter = ax3.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                             c='lightgray', s=1, alpha=0.6)
    
    # Highlight predicted corners
    if len(predicted_corners) > 0:
        ax3.scatter(predicted_corners[:, 0], predicted_corners[:, 1], predicted_corners[:, 2], 
                    c='red', s=100, marker='s', label=f'Predicted Corners (>{threshold})', 
                    edgecolors='black', linewidth=1)
    
    ax3.set_title(f'PointNet2 Predictions (threshold={threshold})')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.legend()
    
    # Add colorbar for probability only when available
    if corner_probs is not None:
        cbar = plt.colorbar(scatter, ax=ax3, shrink=0.5, aspect=20)
        cbar.set_label('Corner Probability')
    
    plt.tight_layout()
    
    # Print detailed statistics
    print(f"\nSample {sample_idx} Results:")
    print(f"Total points: {len(pc_xyz)}")
    print(f"Ground truth corners: {len(gt_corners)}")
    print(f"Predicted corners (>{threshold}): {len(predicted_corners)}")
    if corner_probs is not None:
        print(f"Corner probability range: [{corner_probs.min():.3f}, {corner_probs.max():.3f}]")
        print(f"Average corner probability: {corner_probs.mean():.3f}")
        print(f"Points with prob > 0.1: {(corner_probs > 0.1).sum().item()}")
        print(f"Points with prob > 0.2: {(corner_probs > 0.2).sum().item()}")
        print(f"Points with prob > 0.3: {(corner_probs > 0.3).sum().item()}")
    else:
        if predicted_corners.shape[0] > 0:
            print(f"Predicted corner score range: [{predicted_corner_scores.min():.3f}, {predicted_corner_scores.max():.3f}]")
            print(f"Average predicted corner score: {predicted_corner_scores.mean():.3f}")
        else:
            print("No predicted corners above threshold.")
    
    return fig

def interactive_visualization():
    """Interactive visualization interface"""
    print("PointNet2 Corner Detection Visualization")
    print("=" * 50)
    
    # Get the script directory and construct absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'datasets', 'dataset_config.yaml')
    model_path = os.path.join(project_root, 'output', 'corner_detection_model.pth')
    
    # Load configuration
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return
    
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Fix the root_dir path to be absolute
    if dataset_config.Building3D.root_dir.startswith('./'):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir[2:])
    elif not os.path.isabs(dataset_config.Building3D.root_dir):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir)
    
    # Determine input channels
    input_channels = 3  # xyz coordinates
    if dataset_config.Building3D.use_color:
        input_channels += 4  # rgba
    if dataset_config.Building3D.use_intensity:
        input_channels += 1  # intensity
    if getattr(dataset_config.Building3D, 'use_group_ids', False):
        input_channels += 1  # group_id
    if getattr(dataset_config.Building3D, 'use_border_weights', False):
        input_channels += 1  # border_weights
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running main.py")
        return
    
    print(f"Loading model from {model_path}")
    print(f"Model input channels: {input_channels}")
    model = load_trained_model(model_path, input_channels, device)
    
    # Load dataset
    print("Loading dataset...")
    print(f"Dataset root dir: {dataset_config.Building3D.root_dir}")
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Select dataset split
    print("\nSelect dataset:")
    print("1. Train dataset")
    print("2. Test dataset")
    
    while True:
        split_choice = input("Choose dataset (1 or 2): ").strip()
        if split_choice == "1":
            dataset = building3D_dataset['train']
            split_name = "train"
            break
        elif split_choice == "2":
            dataset = building3D_dataset['test']
            split_name = "test"
            break
        else:
            print("Please enter 1 or 2")
    
    print(f"Selected {split_name} dataset with {len(dataset)} samples")
    
    if len(dataset) == 0:
        print("Error: No samples found in the dataset!")
        print("Please check the dataset path and files.")
        return
    
    # Interactive sample selection
    while True:
        print(f"\nEnter sample index (0-{len(dataset)-1}) or 'q' to quit:")
        print(f"Note: Using threshold ({CORNER_THRESHOLD}) to show confident predictions")
        choice = input("Sample: ").strip()
        
        if choice.lower() == 'q':
            break
            
        try:
            sample_idx = int(choice)
            if 0 <= sample_idx < len(dataset):
                print(f"\nVisualizing sample {sample_idx}...")
                fig = visualize_pointnet2_results(model, dataset, sample_idx, device, threshold=CORNER_THRESHOLD)
                plt.show()
            else:
                print(f"Please enter a number between 0 and {len(dataset)-1}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize PointNet2 corner detection results')
    parser.add_argument('--sample', type=int, default=None, 
                       help='Specific sample index to visualize (if not provided, interactive mode)')
    parser.add_argument('--threshold', type=float, default=CORNER_THRESHOLD,
                       help=f'Threshold for corner predictions (default: {CORNER_THRESHOLD})')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test',
                       help='Dataset split to use (default: test)')
    
    args = parser.parse_args()
    
    if args.sample is not None:
        # Non-interactive mode - visualize specific sample
        print(f"Visualizing sample {args.sample} from {args.split} dataset")
        
        # Get absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        config_path = os.path.join(project_root, 'datasets', 'dataset_config.yaml')
        model_path = os.path.join(project_root, 'output', 'corner_detection_model.pth')
        
        # Load configuration and model
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            return
        
        dataset_config = cfg_from_yaml_file(config_path)
        
        # Fix the root_dir path to be absolute
        if dataset_config.Building3D.root_dir.startswith('./'):
            dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir[2:])
        elif not os.path.isabs(dataset_config.Building3D.root_dir):
            dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir)
        
        input_channels = 3
        if dataset_config.Building3D.use_color:
            input_channels += 4
        if dataset_config.Building3D.use_intensity:
            input_channels += 1
        if getattr(dataset_config.Building3D, 'use_group_ids', False):
            input_channels += 1
        if getattr(dataset_config.Building3D, 'use_border_weights', False):
            input_channels += 1
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        model = load_trained_model(model_path, input_channels, device)
        building3D_dataset = build_dataset(dataset_config.Building3D)
        dataset = building3D_dataset[args.split]
        
        if len(dataset) == 0:
            print(f"Error: No samples found in the {args.split} dataset!")
            return
        
        if args.sample >= len(dataset):
            print(f"Error: Sample index {args.sample} out of range (0-{len(dataset)-1})")
            return
        
        fig = visualize_pointnet2_results(model, dataset, args.sample, device, args.threshold)
        plt.show()
    else:
        # Interactive mode
        interactive_visualization()

if __name__ == "__main__":
    main()