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
    print(f"🔵 Low border weights (≤0.3): {low_border} points ({100*low_border/total:.1f}%)")
    print(f"🟡 Medium border weights (0.3-0.7): {medium_border} points ({100*medium_border/total:.1f}%)")
    print(f"🔴 High border weights (>0.7): {high_border} points ({100*high_border/total:.1f}%)")
    
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
                                  title="Ground Truth vs Predicted Vertices", border_weights=None):
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
        
        mean_error = np.mean(closest_distances)
        max_error = np.max(closest_distances)
        
        print(f"\n📊 Vertex Prediction Accuracy:")
        print(f"   Mean error: {mean_error:.4f}")
        print(f"   Max error: {max_error:.4f}")
        print(f"   Vertices within 0.1 units: {np.sum(closest_distances < 0.1)}/{len(gt_vertices)}")
        
        # Add error metrics to plot
        error_text = f"Mean Error: {mean_error:.4f}\nMax Error: {max_error:.4f}\nAccurate (<0.1): {np.sum(closest_distances < 0.1)}/{len(gt_vertices)}"
        ax.text2D(0.02, 0.98, error_text, transform=ax.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔴 Red circles: Ground truth vertices ({len(gt_vertices)})")
    print(f"🟢 Green triangles: Predicted vertices ({len(pred_vertices)})")
    if gt_edges is not None:
        print(f"🔴 Red lines: Ground truth wireframe ({len(gt_edges)} edges)")
    print(f"💙 Light blue points: Original point cloud")
    
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
        print(f"✅ Loaded trained model from {model_path}")
    else:
        print("⚠️ No saved model found, using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    
    return model, device

def get_sample_data(sample_id=1):
    """Load a specific sample from the dataset"""
    
    print(f"Loading sample {sample_id} from dataset")
    
    # Load dataset
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=4,
        shuffle=False, 
        collate_fn=building3D_dataset['train'].collate_batch
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
            
            print(f"🎯 Found sample {sample_id} in batch {batch_idx + 1}")
            return sample
    
    print(f"❌ Sample {sample_id} not found")
    return None

def predict_vertices_with_model(model, point_cloud, device, max_vertices=50):
    """
    Predict vertex coordinates using the trained model.
    The model directly predicts vertex coordinates and number of vertices.
    """
    with torch.no_grad():
        point_cloud_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0).to(device)
        
        # Get model outputs
        outputs = model(point_cloud_tensor)
        vertex_coords = outputs['vertex_coords'].cpu().numpy()[0]  # [max_vertices, 3]
        num_vertices_sigmoid = outputs['num_vertices_sigmoid'].cpu().numpy()[0]  # sigmoid output
        
        # Convert sigmoid to vertex count
        num_vertices = int(num_vertices_sigmoid * model.max_vertices)  # Scale back to vertex count
        num_vertices = min(max(num_vertices, 1), len(vertex_coords))  # Clamp to valid range
        
        predicted_vertices = vertex_coords[:num_vertices]  # [num_vertices, 3]
        
        print(f"📊 Model predicted {num_vertices} vertices (sigmoid: {num_vertices_sigmoid:.3f})")
        print(f"   Vertex coordinate ranges:")
        print(f"     X: [{predicted_vertices[:, 0].min():.3f}, {predicted_vertices[:, 0].max():.3f}]")
        print(f"     Y: [{predicted_vertices[:, 1].min():.3f}, {predicted_vertices[:, 1].max():.3f}]") 
        print(f"     Z: [{predicted_vertices[:, 2].min():.3f}, {predicted_vertices[:, 2].max():.3f}]")
        
        return predicted_vertices

def main_visualization_matplotlib(sample_id=1, model_path=None):
    """Main function to run matplotlib-based visualization"""
    print("🎨 Building3D Wireframe Reconstruction Visualization (Matplotlib)")
    print("=" * 70)
    
    # Load sample data
    sample = get_sample_data(sample_id)
    if sample is None:
        return
    
    # Extract data
    point_cloud = sample['point_clouds'][0].numpy()  # [N, 5]
    gt_vertices = sample['wf_vertices'][0].numpy()   # [V, 3]
    gt_edges = sample['wf_edges'][0].numpy()         # [E, 2]
    
    # Extract border weights if available
    border_weights = None
    if point_cloud.shape[1] >= 5:
        border_weights = point_cloud[:, 4]  # Last feature is border weight
    
    print(f"\n📊 Sample {sample_id} Data:")
    print(f"   Point cloud: {point_cloud.shape}")
    print(f"   GT vertices: {gt_vertices.shape}")
    print(f"   GT edges: {gt_edges.shape}")
    
    # Load and run model
    model, device = load_trained_model(model_path, input_dim=point_cloud.shape[1])
    predicted_vertices = predict_vertices_with_model(model, point_cloud, device)
    
    # Visualizations
    print(f"\n🎨 Starting matplotlib visualizations...")
    
    # Comparison: Ground truth vs predicted
    print("Ground truth vs predicted vertices...")
    visualize_comparison_matplotlib(
        point_cloud,
        gt_vertices,
        predicted_vertices,
        gt_edges,
        title=f"Sample {sample_id} - GT (Red) vs Predicted (Green) Vertices",
        border_weights=border_weights
    )
    
    print("✅ Matplotlib visualization complete!")

if __name__ == "__main__":
    # Run matplotlib-based visualization for sample 1
    main_visualization_matplotlib(sample_id=1, model_path=None)