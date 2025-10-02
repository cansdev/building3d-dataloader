#!/usr/bin/python3
# _*_ coding: utf-8 _*_ 
# Matplotlib-based visualization for Building3D wireframe reconstruction
# Alternative visualization using matplotlib for broader compatibility

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from datasets import build_dataset
from datasets.building3d import load_wireframe
import yaml
from easydict import EasyDict
from utils.weight_assignment import compute_border_weights
from utils.preprocess import Building3DPreprocessor, create_default_preprocessor
from models.dgcnn_model import BasicDGCNN
from utils.cuda_utils import get_device, to_cuda
from torch.utils.data import DataLoader

def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg

def color_points_by_elevation(points, colormap='viridis'):
    """Color points based on their Z coordinate (elevation)"""
    z_coords = points[:, 2]
    z_normalized = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min()) if z_coords.max() > z_coords.min() else np.ones_like(z_coords) * 0.5
    
    # Use matplotlib colormap (updated syntax)
    try:
        cmap = plt.colormaps[colormap]
    except AttributeError:
        cmap = plt.cm.get_cmap(colormap)  # Fallback for older matplotlib
    colors = cmap(z_normalized)
    
    return colors[:, :3]  # Return only RGB, not alpha

def color_points_by_border_weights(points, border_weights=None, colormap='coolwarm'):
    """Color points based on their border weights"""
    if border_weights is None:
        # Use Z coordinates if no border weights provided
        points_xyz = points[:, :3]
        border_weights = compute_border_weights(points_xyz, k_neighbors=25, normalize_weights=True)
    
    # Normalize weights
    if border_weights.max() > border_weights.min():
        weights_normalized = (border_weights - border_weights.min()) / (border_weights.max() - border_weights.min())
    else:
        weights_normalized = np.ones_like(border_weights) * 0.5
    
    # Use matplotlib colormap (updated syntax)
    try:
        cmap = plt.colormaps[colormap]
    except AttributeError:
        cmap = plt.cm.get_cmap(colormap)  # Fallback for older matplotlib
    colors = cmap(weights_normalized)
    
    # Print statistics
    high_border = np.sum(border_weights > 0.7)
    medium_border = np.sum((border_weights > 0.3) & (border_weights <= 0.7))
    low_border = np.sum(border_weights <= 0.3)
    total = len(border_weights)
    
    print(f"\n=== Border Weight Statistics ===")
    print(f"Low border weights (≤0.3): {low_border} points ({100*low_border/total:.1f}%)")
    print(f"Medium border weights (0.3-0.7): {medium_border} points ({100*medium_border/total:.1f}%)")
    print(f"High border weights (>0.7): {high_border} points ({100*high_border/total:.1f}%)")
    
    return colors[:, :3]  # Return only RGB, not alpha

def visualize_point_cloud_matplotlib(point_cloud, title="Point Cloud", 
                                   color_by='elevation', border_weights=None,
                                   show_coordinate_frame=True, alpha=0.6, point_size=1):
    """Visualize point cloud using matplotlib"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    points = point_cloud[:, :3]
    
    # Color points
    if color_by == 'border_weights' and border_weights is not None:
        colors = color_points_by_border_weights(point_cloud, border_weights, 'coolwarm')
        colorbar_label = 'Border Weight'
    elif color_by == 'elevation':
        colors = color_points_by_elevation(points, 'viridis')
        colorbar_label = 'Elevation (Z)'
    elif point_cloud.shape[1] >= 6:  # Has RGB colors
        colors = point_cloud[:, 3:6]
        if colors.max() > 1.0:  # Normalize if needed
            colors = colors / 255.0
        colorbar_label = 'RGB Color'
    else:
        colors = 'blue'
        colorbar_label = None
    
    # Plot points
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=colors, s=point_size, alpha=alpha)
    
    # Add colorbar if using scalar coloring
    if isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] == 3:
        pass  # RGB colors, no colorbar needed
    elif colorbar_label:
        plt.colorbar(scatter, ax=ax, label=colorbar_label, shrink=0.8)
    
    # Add coordinate frame
    if show_coordinate_frame:
        axis_length = 0.1
        origin = [0, 0, 0]
        
        # X axis (red)
        ax.plot([origin[0], origin[0] + axis_length], [origin[1], origin[1]], [origin[2], origin[2]], 
                'r-', linewidth=3, label='X-axis')
        # Y axis (green)
        ax.plot([origin[0], origin[0]], [origin[1], origin[1] + axis_length], [origin[2], origin[2]], 
                'g-', linewidth=3, label='Y-axis')
        # Z axis (blue)
        ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + axis_length], 
                'b-', linewidth=3, label='Z-axis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make axes equal scale
    max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                         points[:, 1].max()-points[:, 1].min(),
                         points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if show_coordinate_frame:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def visualize_wireframe_matplotlib(vertices, edges, title="Wireframe", 
                                 vertex_color='red', line_color='blue',
                                 vertex_size=50, line_width=1.5, show_coordinate_frame=True):
    """Visualize wireframe using matplotlib"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
              c=vertex_color, s=vertex_size, alpha=0.8, label='Vertices')
    
    # Plot edges
    for edge in edges:
        v1, v2 = int(edge[0]), int(edge[1])  # Ensure indices are integers
        if v1 < len(vertices) and v2 < len(vertices):
            ax.plot([vertices[v1, 0], vertices[v2, 0]], 
                   [vertices[v1, 1], vertices[v2, 1]], 
                   [vertices[v1, 2], vertices[v2, 2]], 
                   color=line_color, linewidth=line_width, alpha=0.7)
    
    # Add coordinate frame
    if show_coordinate_frame:
        axis_length = 0.1
        origin = [0, 0, 0]
        
        # X axis (red)
        ax.plot([origin[0], origin[0] + axis_length], [origin[1], origin[1]], [origin[2], origin[2]], 
                'r-', linewidth=3, label='X-axis')
        # Y axis (green)
        ax.plot([origin[0], origin[0]], [origin[1], origin[1] + axis_length], [origin[2], origin[2]], 
                'g-', linewidth=3, label='Y-axis')
        # Z axis (blue)
        ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + axis_length], 
                'b-', linewidth=3, label='Z-axis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make axes equal scale
    max_range = np.array([vertices[:, 0].max()-vertices[:, 0].min(),
                         vertices[:, 1].max()-vertices[:, 1].min(),
                         vertices[:, 2].max()-vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max()+vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max()+vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max()+vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def visualize_comparison_matplotlib(point_cloud, gt_vertices, pred_vertices, gt_edges=None,
                                  title="Ground Truth vs Predicted Vertices", border_weights=None, max_distance=None):
    """
    Comprehensive comparison visualization using matplotlib
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    print(f"\n=== {title} ===")
    print(f"Point cloud: {point_cloud.shape[0]} points")
    print(f"Ground truth vertices: {gt_vertices.shape[0]} vertices")
    print(f"Predicted vertices: {pred_vertices.shape[0]} vertices")
    
    # Plot point cloud with border weight coloring
    points = point_cloud[:, :3]
    if border_weights is not None:
        point_colors = color_points_by_border_weights(point_cloud, border_weights, 'coolwarm')
    else:
        point_colors = 'lightblue'
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=point_colors, s=0.5, alpha=0.3, label='Point Cloud')
    
    # Plot ground truth vertices (red)
    ax.scatter(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2], 
              c='red', s=100, alpha=0.9, marker='o', label='GT Vertices', edgecolors='darkred')
    
    # Plot predicted vertices (green)
    ax.scatter(pred_vertices[:, 0], pred_vertices[:, 1], pred_vertices[:, 2], 
              c='lime', s=60, alpha=0.9, marker='^', label='Pred Vertices', edgecolors='darkgreen')
    
    # Plot ground truth wireframe edges
    if gt_edges is not None:
        for edge in gt_edges:
            v1, v2 = int(edge[0]), int(edge[1])  # Ensure indices are integers
            if v1 < len(gt_vertices) and v2 < len(gt_vertices):
                ax.plot([gt_vertices[v1, 0], gt_vertices[v2, 0]], 
                       [gt_vertices[v1, 1], gt_vertices[v2, 1]], 
                       [gt_vertices[v1, 2], gt_vertices[v2, 2]], 
                       color='red', linewidth=1.5, alpha=0.6)
    
    # Add coordinate frame
    axis_length = 0.1
    origin = [0, 0, 0]
    
    # X axis (red)
    ax.plot([origin[0], origin[0] + axis_length], [origin[1], origin[1]], [origin[2], origin[2]], 
            'r-', linewidth=4, label='X-axis')
    # Y axis (green)
    ax.plot([origin[0], origin[0]], [origin[1], origin[1] + axis_length], [origin[2], origin[2]], 
            'g-', linewidth=4, label='Y-axis')
    # Z axis (blue)
    ax.plot([origin[0], origin[0]], [origin[1], origin[1]], [origin[2], origin[2] + axis_length], 
            'b-', linewidth=4, label='Z-axis')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Make axes equal scale
    all_points = np.vstack([points, gt_vertices, pred_vertices])
    max_range = np.array([all_points[:, 0].max()-all_points[:, 0].min(),
                         all_points[:, 1].max()-all_points[:, 1].min(),
                         all_points[:, 2].max()-all_points[:, 2].min()]).max() / 2.0
    mid_x = (all_points[:, 0].max()+all_points[:, 0].min()) * 0.5
    mid_y = (all_points[:, 1].max()+all_points[:, 1].min()) * 0.5
    mid_z = (all_points[:, 2].max()+all_points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Compute error metrics
    if len(pred_vertices) > 0 and len(gt_vertices) > 0:
        # Find closest predicted vertex for each ground truth vertex
        from scipy.spatial.distance import cdist
        distances = cdist(gt_vertices, pred_vertices)
        closest_pred_indices = np.argmin(distances, axis=1)
        closest_distances = np.min(distances, axis=1)
        
        mean_error_norm = np.mean(closest_distances)
        max_error_norm = np.max(closest_distances)
        
        # Denormalize errors to real-world scale
        if max_distance is not None:
            mean_error_world = mean_error_norm * max_distance
            max_error_world = max_error_norm * max_distance
            threshold_norm = 0.05 / max_distance  # 50mm threshold in normalized space
            accurate_count = np.sum(closest_distances < threshold_norm)
            
            print(f"\nVertex Prediction Accuracy:")
            print(f"   Mean error (normalized): {mean_error_norm:.4f}")
            print(f"   Mean error (real-world): {mean_error_world:.4f}m ({mean_error_world*1000:.0f}mm)")
            print(f"   Max error (real-world): {max_error_world:.4f}m ({max_error_world*1000:.0f}mm)")
            print(f"   Vertices within 50mm: {accurate_count}/{len(gt_vertices)}")
            
            # Add error metrics to plot
            error_text = (f"Mean Error: {mean_error_world:.4f}m ({mean_error_world*1000:.0f}mm)\n"
                         f"Max Error: {max_error_world:.4f}m ({max_error_world*1000:.0f}mm)\n"
                         f"Accurate (<50mm): {accurate_count}/{len(gt_vertices)}")
        else:
            print(f"\nVertex Prediction Accuracy (normalized):")
            print(f"   Mean error: {mean_error_norm:.4f}")
            print(f"   Max error: {max_error_norm:.4f}")
            print(f"   Vertices within 0.1 units: {np.sum(closest_distances < 0.1)}/{len(gt_vertices)}")
            
            # Add error metrics to plot
            error_text = f"Mean Error: {mean_error_norm:.4f}\nMax Error: {max_error_norm:.4f}\nAccurate (<0.1): {np.sum(closest_distances < 0.1)}/{len(gt_vertices)}"
        
        ax.text2D(0.02, 0.98, error_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    print(f"\nRed circles: Ground truth vertices ({len(gt_vertices)})")
    print(f"Green triangles: Predicted vertices ({len(pred_vertices)})")
    if gt_edges is not None:
        print(f"Red lines: Ground truth wireframe ({len(gt_edges)} edges)")
    print(f"Light blue points: Original point cloud")
    
    return fig, ax

def load_trained_model(model_path=None, input_dim=5, k=20, device=None):
    """Load a trained DGCNN model"""
    if device is None:
        device = get_device()
    
    model = BasicDGCNN(input_dim=input_dim, k=k)
    
    # Try to load the saved model
    if model_path is None:
        model_path = 'trained_dgcnn_model.pth'  # Default path
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded trained model from {model_path}")
    else:
        print("Warning: No saved model found, using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def get_sample_data(sample_id=1, use_test_split=True):
    """
    Load a specific sample from the dataset
    
    Args:
        sample_id: ID of the sample to load
        use_test_split: If True, load from test split (no augmentation)
                       If False, load from train split (with augmentation if enabled)
    """
    
    print(f"Loading sample {sample_id} from dataset ({'test' if use_test_split else 'train'} split)")
    
    # Load dataset
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    
    # IMPORTANT: Temporarily disable augmentation AND preprocessing cache for visualization
    # We want to see the ORIGINAL raw data
    original_augment = dataset_config.Building3D.augment
    dataset_config.Building3D.augment = False  # Force disable augmentation
    
    # Also disable preprocessing cache to ensure fresh load
    original_use_cache = dataset_config.Building3D.preprocessor.get('use_cache', True)
    if 'preprocessor' not in dataset_config.Building3D:
        dataset_config.Building3D.preprocessor = {}
    dataset_config.Building3D.preprocessor.use_cache = False  # Force fresh preprocessing
    
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Restore cache setting
    dataset_config.Building3D.preprocessor.use_cache = original_use_cache
    
    # Choose split (test has no augmentation, train might have)
    dataset_split = building3D_dataset['test'] if use_test_split else building3D_dataset['train']
    
    train_loader = DataLoader(
        dataset_split, 
        batch_size=4,
        shuffle=False, 
        collate_fn=dataset_split.collate_batch
    )
    
    # Find the sample
    for batch_idx, batch_data in enumerate(train_loader):
        if 'scan_idx' in batch_data and sample_id in batch_data['scan_idx']:
            sample_pos = (batch_data['scan_idx'] == sample_id).nonzero(as_tuple=True)[0][0]
            
            sample = {}
            for key, value in batch_data.items():
                if isinstance(value, torch.Tensor) and len(value.shape) > 0:
                    sample[key] = value[sample_pos:sample_pos+1]
                else:
                    sample[key] = value
            
            print(f"Found sample {sample_id} in batch {batch_idx + 1}")
            print(f"   Augmentation was: {'ENABLED' if original_augment else 'DISABLED'} in config")
            print(f"   Visualization: Showing ORIGINAL (non-augmented) data")
            return sample
    
    print(f"Sample {sample_id} not found")
    return None

def predict_vertices_with_model(model, point_cloud, device, max_vertices=50):
    """
    Predict vertex coordinates using the trained model.
    The model now uses per-vertex existence prediction instead of global count.
    """
    with torch.no_grad():
        point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
        
        # Get model outputs
        outputs = model(point_cloud_tensor)
        vertex_coords = outputs['vertex_coords'].cpu().numpy()[0]  # [max_vertices, 3]
        existence_probs = outputs['existence_probs'].cpu().numpy()[0]  # [max_vertices]
        num_vertices = outputs['num_vertices'].cpu().numpy()[0]  # Dynamic count
        
        # Filter vertices based on existence probability threshold (0.5)
        vertex_exists = existence_probs > 0.5
        predicted_vertices = vertex_coords[vertex_exists]  # [num_existing_vertices, 3]
        
        # Fallback: if no vertices pass threshold, use top-k by probability
        if len(predicted_vertices) == 0:
            num_vertices = max(1, int(num_vertices))
            top_k_indices = existence_probs.argsort()[::-1][:num_vertices]
            predicted_vertices = vertex_coords[top_k_indices]
        
        print(f"Model predicted {len(predicted_vertices)} vertices")
        print(f"   Existence probabilities: min={existence_probs.min():.3f}, max={existence_probs.max():.3f}, mean={existence_probs.mean():.3f}")
        print(f"   Vertices with prob>0.5: {vertex_exists.sum()}")
        print(f"   Vertex coordinate ranges:")
        print(f"     X: [{predicted_vertices[:, 0].min():.3f}, {predicted_vertices[:, 0].max():.3f}]")
        print(f"     Y: [{predicted_vertices[:, 1].min():.3f}, {predicted_vertices[:, 1].max():.3f}]") 
        print(f"     Z: [{predicted_vertices[:, 2].min():.3f}, {predicted_vertices[:, 2].max():.3f}]")
        
        return predicted_vertices

def main_visualization_matplotlib(sample_id=1, model_path=None):
    """Main function to run matplotlib-based visualization"""
    print("Building3D Wireframe Reconstruction Visualization (Matplotlib)")
    print("=" * 70)
    
    # Load sample data from train split (Sample 1 is in train split)
    sample = get_sample_data(sample_id, use_test_split=False)
    if sample is None:
        return
    
    # Extract data
    point_cloud = sample['point_clouds'][0].numpy()  # [N, 5]
    gt_vertices_raw = sample['wf_vertices'][0].numpy()   # [V, 3] with padding
    gt_edges_raw = sample['wf_edges'][0].numpy()         # [E, 2] with padding
    
    # Extract normalization scale factor
    max_distance = None
    if 'max_distance' in sample:
        max_distance = sample['max_distance'][0].item() if torch.is_tensor(sample['max_distance']) else sample['max_distance']
    
    # Filter out padded vertices (padding uses -10.0)
    valid_vertex_mask = gt_vertices_raw[:, 0] > -9.0
    gt_vertices = gt_vertices_raw[valid_vertex_mask]
    
    # Filter out padded edges (padding uses -1)
    valid_edge_mask = gt_edges_raw[:, 0] > -0.5
    gt_edges = gt_edges_raw[valid_edge_mask]
    
    print(f"Filtered padding: {np.sum(valid_vertex_mask)} real vertices (removed {np.sum(~valid_vertex_mask)} padded)")
    print(f"Filtered padding: {np.sum(valid_edge_mask)} real edges (removed {np.sum(~valid_edge_mask)} padded)")
    
    # Extract border weights if available
    border_weights = None
    if point_cloud.shape[1] >= 5:
        border_weights = point_cloud[:, 4]  # Last feature is border weight
    
    print(f"\nSample {sample_id} Data:")
    print(f"   Point cloud: {point_cloud.shape}")
    print(f"   GT vertices: {gt_vertices.shape}")
    print(f"   GT edges: {gt_edges.shape}")
    if max_distance is not None:
        print(f"   Normalization scale: {max_distance:.2f}m (used for denormalizing errors)")
    print(f"   GT vertex coordinate ranges (original orientation):")
    print(f"     X: [{gt_vertices[:, 0].min():.3f}, {gt_vertices[:, 0].max():.3f}]")
    print(f"     Y: [{gt_vertices[:, 1].min():.3f}, {gt_vertices[:, 1].max():.3f}]")
    print(f"     Z: [{gt_vertices[:, 2].min():.3f}, {gt_vertices[:, 2].max():.3f}]")
    
    # ===== OPTION 2: Transform GT to canonical space =====
    # The model predicts in canonical space, so we need to transform GT to canonical too
    from models.dgcnn_model import compute_pca_alignment
    coords_tensor = torch.from_numpy(point_cloud[:, :3]).unsqueeze(0).float()  # [1, N, 3]
    with torch.no_grad():
        _, _, rotation_matrix, centroid = compute_pca_alignment(coords_tensor, return_transform=True)
    
    # Transform GT to canonical space
    gt_vertices_tensor = torch.from_numpy(gt_vertices).float()
    gt_centered = gt_vertices_tensor - centroid[0].cpu()
    gt_vertices_canonical = torch.matmul(gt_centered, rotation_matrix[0].cpu()).numpy()
    
    # Transform point cloud to canonical space too (for visualization consistency)
    point_cloud_coords = torch.from_numpy(point_cloud[:, :3]).float()
    pc_centered = point_cloud_coords - centroid[0].cpu()
    point_cloud_canonical = torch.matmul(pc_centered, rotation_matrix[0].cpu()).numpy()
    
    # Reconstruct full point cloud with transformed coordinates
    point_cloud_canonical_full = np.copy(point_cloud)
    point_cloud_canonical_full[:, :3] = point_cloud_canonical
    
    print(f"   GT and point cloud transformed to canonical space for comparison")
    print(f"   GT vertex coordinate ranges (canonical space):")
    print(f"     X: [{gt_vertices_canonical[:, 0].min():.3f}, {gt_vertices_canonical[:, 0].max():.3f}]")
    print(f"     Y: [{gt_vertices_canonical[:, 1].min():.3f}, {gt_vertices_canonical[:, 1].max():.3f}]")
    print(f"     Z: [{gt_vertices_canonical[:, 2].min():.3f}, {gt_vertices_canonical[:, 2].max():.3f}]")
    
    # Load and run model
    model, device = load_trained_model(model_path, input_dim=point_cloud.shape[1])
    predicted_vertices = predict_vertices_with_model(model, point_cloud, device)
    
    # Visualizations
    print(f"\nStarting matplotlib visualizations...")
    
    # Comparison: Ground truth vs predicted (ALL IN CANONICAL SPACE)
    print("Ground truth vs predicted vertices (in canonical space)...")
    visualize_comparison_matplotlib(
        point_cloud_canonical_full,  # Use canonical point cloud (aligned)
        gt_vertices_canonical,       # Use canonical GT
        predicted_vertices,          # Already in canonical space
        gt_edges,
        title=f"Sample {sample_id} - GT (Red) vs Predicted (Green) Vertices (Canonical Space)",
        border_weights=border_weights,
        max_distance=max_distance
    )
    
    print("Matplotlib visualization complete!")

if __name__ == "__main__":
    # Run matplotlib-based visualization for sample 1 (from train split)
    # Note: Sample 1 is in train split, so we use use_test_split=False
    main_visualization_matplotlib(sample_id=1004, model_path=None)