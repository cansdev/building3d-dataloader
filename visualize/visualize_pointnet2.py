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
import glob
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset
from datasets.building3d import calculate_input_dim
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

def detect_model_files(project_root):
    """Detect all .pth model files in the output directory"""
    output_dir = os.path.join(project_root, 'output')
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Find all .pth files in output directory
    pth_files = glob.glob(os.path.join(output_dir, '*.pth'))
    
    if not pth_files:
        raise FileNotFoundError(f"No .pth model files found in {output_dir}")
    
    # Sort files by modification time (newest first)
    pth_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return pth_files

def choose_model_file(pth_files):
    """Interactive model file selection"""
    print("\nAvailable model files:")
    print("=" * 80)
    
    for i, pth_file in enumerate(pth_files):
        filename = os.path.basename(pth_file)
        # Get file modification time
        mod_time = os.path.getmtime(pth_file)
        mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
        # Get file size
        file_size = os.path.getsize(pth_file)
        size_mb = file_size / (1024 * 1024)
        
        print(f"\n{i}: {filename}")
        print(f"   Modified: {mod_time_str} | Size: {size_mb:.1f} MB")
        
        # Try to load and display configuration
        try:
            checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    print(f"   Config: color={config.get('use_color', '?')}, "
                          f"intensity={config.get('use_intensity', '?')}, "
                          f"group_ids={config.get('use_group_ids', '?')}, "
                          f"border_weights={config.get('use_border_weights', '?')}")
                    print(f"   Channels: {config.get('input_channels', '?')} | "
                          f"Points: {config.get('num_points', '?')}")
                    
                    # Display border attention weight if available
                    if 'border_attention_weight' in checkpoint:
                        raw_w = checkpoint['border_attention_weight']
                        # Calculate effective constrained weight
                        import math
                        constrained_w = 0.12 / (1.0 + math.exp(-raw_w))  # 0.12 * sigmoid (max 12% boost)
                        print(f"   Border attention: raw={raw_w:.4f}, effective={constrained_w:.4f} (max_boost=12%)")
                        
                elif 'input_channels' in checkpoint:
                    print(f"   Channels: {checkpoint['input_channels']} (config details not saved)")
                
                if 'epoch' in checkpoint:
                    print(f"   Epoch: {checkpoint['epoch']}")
                if 'loss' in checkpoint:
                    print(f"   Loss: {checkpoint['loss']:.4f}")
        except Exception as e:
            print(f"   (Could not read model metadata)")
    
    print("\n" + "=" * 80)
    
    while True:
        try:
            choice = input(f"Choose model file (0-{len(pth_files)-1}): ").strip()
            model_idx = int(choice)
            
            if 0 <= model_idx < len(pth_files):
                selected_file = pth_files[model_idx]
                print(f"\n✓ Selected: {os.path.basename(selected_file)}")
                return selected_file
            else:
                print(f"Please enter a number between 0 and {len(pth_files)-1}")
        except ValueError:
            print("Please enter a valid number")

def setup_environment_non_interactive(model_path=None):
    """Setup environment for non-interactive mode with optional model path"""
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'datasets', 'dataset_config.yaml')
    
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Fix the root_dir path to be absolute
    if dataset_config.Building3D.root_dir.startswith('./'):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir[2:])
    elif not os.path.isabs(dataset_config.Building3D.root_dir):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir)
    
    # Fix the cache_dir path to be absolute and point to root-level preprocessed folder
    if hasattr(dataset_config.Building3D, 'preprocessor') and hasattr(dataset_config.Building3D.preprocessor, 'cache_dir'):
        cache_dir = dataset_config.Building3D.preprocessor.cache_dir
        if cache_dir.startswith('./'):
            dataset_config.Building3D.preprocessor.cache_dir = os.path.join(project_root, cache_dir[2:])
        elif not os.path.isabs(cache_dir):
            dataset_config.Building3D.preprocessor.cache_dir = os.path.join(project_root, cache_dir)
    
    # Use the dataloader's calculate_input_dim function
    input_channels = calculate_input_dim(dataset_config.Building3D)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle model path
    if model_path is None:
        # Use default model path for backward compatibility
        model_path = os.path.join(project_root, 'output', 'corner_detection_model.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Default model file not found at {model_path}")
    else:
        # Use provided model path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load dataset
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    return {
        'config': dataset_config,
        'input_channels': input_channels,
        'device': device,
        'model_path': model_path,
        'datasets': building3D_dataset
    }

def setup_environment():
    """Setup environment for interactive mode with model selection"""
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    config_path = os.path.join(project_root, 'datasets', 'dataset_config.yaml')
    
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Fix the root_dir path to be absolute
    if dataset_config.Building3D.root_dir.startswith('./'):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir[2:])
    elif not os.path.isabs(dataset_config.Building3D.root_dir):
        dataset_config.Building3D.root_dir = os.path.join(project_root, dataset_config.Building3D.root_dir)
    
    # Fix the cache_dir path to be absolute and point to root-level preprocessed folder
    if hasattr(dataset_config.Building3D, 'preprocessor') and hasattr(dataset_config.Building3D.preprocessor, 'cache_dir'):
        cache_dir = dataset_config.Building3D.preprocessor.cache_dir
        if cache_dir.startswith('./'):
            dataset_config.Building3D.preprocessor.cache_dir = os.path.join(project_root, cache_dir[2:])
        elif not os.path.isabs(cache_dir):
            dataset_config.Building3D.preprocessor.cache_dir = os.path.join(project_root, cache_dir)
    
    # Use the dataloader's calculate_input_dim function
    input_channels = calculate_input_dim(dataset_config.Building3D)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Detect and choose model file
    try:
        pth_files = detect_model_files(project_root)
        model_path = choose_model_file(pth_files)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}\nPlease train the model first by running train.py")
    
    # Load dataset
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    return {
        'config': dataset_config,
        'input_channels': input_channels,
        'device': device,
        'model_path': model_path,
        'datasets': building3D_dataset
    }

def load_trained_model(model_path, input_channels, device):
    """Load trained PointNet2 model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle both checkpoint formats: full checkpoint dict or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format: checkpoint contains model_state_dict and metadata
        saved_input_channels = checkpoint.get('input_channels', None)
        has_border_weights = False
        
        # Display saved configuration if available
        if 'config' in checkpoint:
            config = checkpoint['config']
            has_border_weights = config.get('use_border_weights', False)
            
            print("\n" + "=" * 50)
            print("MODEL CONFIGURATION (from checkpoint)")
            print("=" * 50)
            print(f"  use_color: {config.get('use_color', 'N/A')}")
            print(f"  use_intensity: {config.get('use_intensity', 'N/A')}")
            print(f"  use_group_ids: {config.get('use_group_ids', 'N/A')}")
            print(f"  use_border_weights: {has_border_weights}")
            print(f"  normalize: {config.get('normalize', 'N/A')}")
            print(f"  num_points: {config.get('num_points', 'N/A')}")
            print(f"  input_channels: {config.get('input_channels', 'N/A')}")
            
            # Display border attention weight if available
            if 'border_attention_weight' in checkpoint:
                raw_w = checkpoint['border_attention_weight']
                # Calculate effective constrained weight
                import math
                constrained_w = 0.12 / (1.0 + math.exp(-raw_w))  # 0.12 * sigmoid (max 12% boost)
                print(f"  border_attention: raw={raw_w:.4f}, effective={constrained_w:.4f} (max_boost=12%)")
            
            print("=" * 50)
        
        if saved_input_channels is not None and saved_input_channels != input_channels:
            print(f"⚠️  WARNING: Model was trained with {saved_input_channels} input channels, "
                  f"but config specifies {input_channels} channels!")
            print(f"    Using the model's trained architecture ({saved_input_channels} channels).")
            input_channels = saved_input_channels
        
        # Use saved config for model creation
        model = PointNet2CornerDetection(config=config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'epoch' in checkpoint:
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Model architecture: {input_channels} input channels, border_attention={has_border_weights}")
    else:
        # Old format: checkpoint is just the state_dict
        print("⚠️  WARNING: Old checkpoint format detected. Cannot verify input channels or configuration.")
        print(f"    Assuming {input_channels} input channels from config.")
        
        # Create minimal config for old models
        minimal_config = {'input_channels': input_channels, 'use_border_weights': False}
        model = PointNet2CornerDetection(config=minimal_config)
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

from matplotlib.widgets import Slider, Button
import matplotlib
# Try to use interactive backend
try:
    matplotlib.use('TkAgg')
except:
    pass

def visualize_pointnet2_results(model, dataset, sample_idx, device, threshold=CORNER_THRESHOLD):
    """
    Visualize PointNet2 corner detection results for a specific sample with interactive threshold slider
    """
    sample = dataset[sample_idx]
    if isinstance(sample['point_clouds'], np.ndarray):
        point_clouds = torch.from_numpy(sample['point_clouds']).unsqueeze(0).to(device)
    else:
        point_clouds = sample['point_clouds'].unsqueeze(0).to(device)
    if isinstance(sample['wf_vertices'], np.ndarray):
        wf_vertices = torch.from_numpy(sample['wf_vertices']).unsqueeze(0).to(device)
    else:
        wf_vertices = sample['wf_vertices'].unsqueeze(0).to(device)
    valid_mask = wf_vertices[0, :, 0] > -1e0
    gt_corners = wf_vertices[0, valid_mask].cpu().numpy()

    with torch.no_grad():
        outputs = model(point_clouds)
        if isinstance(outputs, dict) and 'pred_logits' in outputs and 'pred_boxes' in outputs:
            batch_corners, batch_scores = model.get_corner_predictions(point_clouds, threshold=threshold)
            predicted_corner_coords = batch_corners[0].cpu().numpy()
            predicted_corner_scores = batch_scores[0].cpu().numpy()
            corner_probs = None
            corner_logits = None
        else:
            corner_logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
            corner_probs = torch.sigmoid(corner_logits[0])

    pc_xyz = point_clouds[0, :, :3].cpu().numpy()

    # Initial threshold value
    threshold_val = threshold

    # Create visualization with proper spacing for slider
    fig = plt.figure(figsize=(16, 8))
    
    # Create subplot grid: 2 columns for 2 plots, leave space at bottom for slider
    gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], hspace=0.3, top=0.95, bottom=0.1, wspace=0.3)
    
    ax2 = fig.add_subplot(gs[0, 0], projection='3d')
    ax3 = fig.add_subplot(gs[0, 1], projection='3d')

    # Plot 1: Ground truth corners
    ax2.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c='lightblue', s=1, alpha=0.3, label='Point Cloud')
    if len(gt_corners) > 0:
        ax2.scatter(gt_corners[:, 0], gt_corners[:, 1], gt_corners[:, 2], c='red', s=50, marker='o', label='GT Corners')
    ax2.set_title('Ground Truth Corners')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    # Store colorbar reference
    colorbar_obj = [None]
    
    # Plot 2: Predicted corners (will be updated by slider)
    def update_plot(threshold):
        ax3.clear()
        
        # Plot point cloud with probabilities
        if corner_probs is not None:
            scatter = ax3.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                                 c=corner_probs.cpu().numpy(), cmap='viridis', 
                                 s=2, alpha=0.8, vmin=0.0, vmax=1.0)
            corner_predictions = (corner_probs > threshold)
            predicted_corners = pc_xyz[corner_predictions.cpu().numpy()]
            
            # Update colorbar if it doesn't exist
            if colorbar_obj[0] is None:
                colorbar_obj[0] = plt.colorbar(scatter, ax=ax3, shrink=0.5, aspect=20)
                colorbar_obj[0].set_label('Corner Probability')
        else:
            scatter = ax3.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], 
                                 c='lightgray', s=1, alpha=0.6)
            if 'predicted_corner_coords' in locals():
                predicted_corners = predicted_corner_coords
            else:
                predicted_corners = np.empty((0, 3))
        
        # Plot predicted corners above threshold
        if len(predicted_corners) > 0:
            ax3.scatter(predicted_corners[:, 0], predicted_corners[:, 1], predicted_corners[:, 2],
                        c='red', s=100, marker='s', label=f'Predicted Corners (>{threshold:.2f})',
                        edgecolors='black', linewidth=1)
        
        ax3.set_title(f'PointNet2 Predictions (threshold={threshold:.2f})', fontsize=12, pad=10)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        # Print stats to console
        if corner_probs is not None:
            n_predicted = (corner_probs > threshold).sum().item()
            print(f"Threshold {threshold:.2f}: {n_predicted} predicted corners")
        
        fig.canvas.draw_idle()

    # Initial plot
    update_plot(threshold_val)

    # Create slider axes in the bottom row with better positioning
    slider_ax = fig.add_subplot(gs[1, :])
    slider_ax.set_position([0.15, 0.03, 0.7, 0.02])  # [left, bottom, width, height]
    slider_ax.set_facecolor('lightgray')
    
    # Create slider with better visual feedback
    threshold_slider = Slider(
        ax=slider_ax,
        label='Threshold (use arrow keys or drag)',
        valmin=0.0,
        valmax=1.0,
        valinit=threshold_val,
        valstep=0.01,
        color='steelblue',
        track_color='lightgray'
    )

    # Connect slider to update function
    def on_slider_change(val):
        update_plot(val)

    threshold_slider.on_changed(on_slider_change)
    
    # Add keyboard controls for easier threshold adjustment
    def on_key_press(event):
        current_val = threshold_slider.val
        step = 0.05  # 5% increments
        
        if event.key == 'right' or event.key == 'up':
            new_val = min(1.0, current_val + step)
            threshold_slider.set_val(new_val)
        elif event.key == 'left' or event.key == 'down':
            new_val = max(0.0, current_val - step)
            threshold_slider.set_val(new_val)
        elif event.key == 'r':
            # Reset to initial threshold
            threshold_slider.set_val(threshold_val)
    
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Add instruction text
    fig.text(0.5, 0.005, 'Controls: Drag slider OR use Arrow Keys (←/→) to adjust | R to reset | Q to close', 
             ha='center', fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Print detailed statistics (initial)
    print(f"\nSample {sample_idx} Results:")
    print(f"Total points: {len(pc_xyz)}")
    print(f"Ground truth corners: {len(gt_corners)}")
    if corner_probs is not None:
        print(f"Corner probability range: [{corner_probs.min():.3f}, {corner_probs.max():.3f}]")
        print(f"Average corner probability: {corner_probs.mean():.3f}")
        print(f"Points with prob > 0.1: {(corner_probs > 0.1).sum().item()}")
        print(f"Points with prob > 0.2: {(corner_probs > 0.2).sum().item()}")
        print(f"Points with prob > 0.3: {(corner_probs > 0.3).sum().item()}")
        print(f"Points with prob > 0.4: {(corner_probs > 0.4).sum().item()}")
        print(f"Points with prob > 0.5: {(corner_probs > 0.5).sum().item()}")
        print(f"Points with prob > 0.6: {(corner_probs > 0.6).sum().item()}")
        print(f"Points with prob > 0.7: {(corner_probs > 0.7).sum().item()}")

    return fig

def interactive_visualization():
    """Interactive visualization interface"""
    print("PointNet2 Corner Detection Visualization")
    print("=" * 50)
    
    try:
        # Setup environment using helper function
        env = setup_environment()
        
        print(f"Using device: {env['device']}")
        print(f"Loading model from {env['model_path']}")
        print(f"Model input channels: {env['input_channels']}")
        
        # Load model
        model = load_trained_model(env['model_path'], env['input_channels'], env['device'])
        
        # Load dataset
        print("Loading dataset...")
        print(f"Dataset root dir: {env['config'].Building3D.root_dir}")
        
    except (FileNotFoundError, Exception) as e:
        print(f"Error: {e}")
        if "Model file not found" in str(e):
            print("Please train the model first by running train.py")
        return
    
    # Select dataset split
    print("\nSelect dataset:")
    print("1. Train dataset")
    print("2. Test dataset")
    
    while True:
        split_choice = input("Choose dataset (1 or 2): ").strip()
        if split_choice == "1":
            dataset = env['datasets']['train']
            split_name = "train"
            break
        elif split_choice == "2":
            dataset = env['datasets']['test']
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
                fig = visualize_pointnet2_results(model, dataset, sample_idx, env['device'], threshold=CORNER_THRESHOLD)
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
    parser.add_argument('--model', type=str, default=None,
                       help='Path to specific model file (if not provided, uses default or interactive selection)')
    
    args = parser.parse_args()
    
    if args.sample is not None:
        # Non-interactive mode - visualize specific sample
        print(f"Visualizing sample {args.sample} from {args.split} dataset")
        
        try:
            # Setup environment using helper function
            env = setup_environment_non_interactive(model_path=args.model)
            
            # Load model
            model = load_trained_model(env['model_path'], env['input_channels'], env['device'])
            dataset = env['datasets'][args.split]
            
            if len(dataset) == 0:
                print(f"Error: No samples found in the {args.split} dataset!")
                return
            
            if args.sample >= len(dataset):
                print(f"Error: Sample index {args.sample} out of range (0-{len(dataset)-1})")
                return
            
            fig = visualize_pointnet2_results(model, dataset, args.sample, env['device'], args.threshold)
            plt.show()
            
        except (FileNotFoundError, Exception) as e:
            print(f"Error: {e}")
            return
    else:
        # Interactive mode
        interactive_visualization()

if __name__ == "__main__":
    main()