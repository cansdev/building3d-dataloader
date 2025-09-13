import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset
from datasets.building3d import load_wireframe
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

def visualize_point_cloud_matplotlib(point_cloud, title="Point Cloud", max_points=1000):
    """Visualize point cloud using matplotlib"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
    else:
        sampled_points = point_cloud
    
    # Extract coordinates
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    
    # Use colors if available
    if sampled_points.shape[1] > 6:  # Has RGB
        colors = sampled_points[:, 3:6]
        if colors.max() > 1.0:  # Normalize if needed
            colors = colors / 255.0
        ax.scatter(x, y, z, c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(x, y, z, c='blue', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    return fig, ax

def visualize_point_cloud_open3d(point_cloud, title="Point Cloud"):
    """Visualize point cloud using Open3D"""
    try:
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Add colors if available
        if point_cloud.shape[1] > 6:
            colors = point_cloud[:, 3:6]
            if colors.max() > 1.0:  # Normalize if needed
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        print(f"Displaying {title} using Open3D...")
        o3d.visualization.draw_geometries([pcd], window_name=title)
        
        return pcd
        
    except ImportError:
        print("Open3D not available. Install with: pip install open3d")
        return None

def visualize_wireframe_matplotlib(vertices, edges, title="Wireframe", color='blue'):
    """Visualize wireframe using matplotlib"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    # Plot edges
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax.plot(x_vals, y_vals, z_vals, color=color, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

def visualize_wireframe_open3d(vertices, edges, title="Wireframe"):
    """Visualize wireframe using Open3D"""
    try:
        import open3d as o3d
        
        # Create line set for wireframe
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        
        # Create point cloud for vertices
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.paint_uniform_color([1, 0, 0])  # Red vertices
        
        # Color lines blue
        colors = [[0, 0, 1] for _ in range(len(edges))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # Visualize
        print(f"Displaying {title} using Open3D...")
        o3d.visualization.draw_geometries([line_set, pcd], window_name=title)
        
        return line_set, pcd
        
    except ImportError:
        print("Open3D not available. Install with: pip install open3d")
        return None, None

def visualize_combined_matplotlib(point_cloud, vertices, edges, title="Combined View"):
    """Visualize point cloud and wireframe together using matplotlib"""
    fig = plt.figure(figsize=(15, 10))
    
    # Point cloud subplot
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Sample points if too many
    max_points = 1000
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
    else:
        sampled_points = point_cloud
    
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    
    if sampled_points.shape[1] > 6:
        colors = sampled_points[:, 3:6]
        if colors.max() > 1.0:
            colors = colors / 255.0
        ax1.scatter(x, y, z, c=colors, s=1, alpha=0.6)
    else:
        ax1.scatter(x, y, z, c='lightblue', s=1, alpha=0.6)
    
    ax1.set_title('Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Wireframe subplot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot vertices
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=50, alpha=0.8, label='Vertices')
    
    # Plot edges
    for edge in edges:
        v1, v2 = edge[0], edge[1]
        x_vals = [vertices[v1, 0], vertices[v2, 0]]
        y_vals = [vertices[v1, 1], vertices[v2, 1]]
        z_vals = [vertices[v1, 2], vertices[v2, 2]]
        ax2.plot(x_vals, y_vals, z_vals, color='blue', linewidth=2, alpha=0.7)
    
    ax2.set_title('Wireframe')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def visualize_combined_open3d(point_cloud, vertices, edges, title="Combined View"):
    """Visualize point cloud and wireframe together using Open3D"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        if point_cloud.shape[1] > 6:
            colors = point_cloud[:, 3:6]
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.7, 0.7, 1.0])  # Light blue
        
        # Create wireframe
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        
        # Create vertex points
        vertex_pcd = o3d.geometry.PointCloud()
        vertex_pcd.points = o3d.utility.Vector3dVector(vertices)
        vertex_pcd.paint_uniform_color([1, 0, 0])  # Red vertices
        
        # Color wireframe edges
        edge_colors = [[0, 0, 1] for _ in range(len(edges))]
        line_set.colors = o3d.utility.Vector3dVector(edge_colors)
        
        # Visualize all together
        print(f"Displaying {title} using Open3D...")
        o3d.visualization.draw_geometries([pcd, line_set, vertex_pcd], window_name=title)
        
        return pcd, line_set, vertex_pcd
        
    except ImportError:
        print("Open3D not available. Install with: pip install open3d")
        return None, None, None

def select_sample(dataset):
    """Interactive sample selection"""
    print(f"\nAvailable samples: {len(dataset)}")
    print("Sample files:")
    
    # Show available samples
    for i in range(min(10, len(dataset))):  # Show first 10 samples
        sample_name = os.path.basename(dataset.pc_files[i]).replace('.xyz', '')
        print(f"  {i+1}: {sample_name}")
    
    if len(dataset) > 10:
        print(f"  ... and {len(dataset) - 10} more samples")
    
    while True:
        try:
            choice = input(f"\nEnter sample number (1-{len(dataset)}): ").strip()
            sample_idx = int(choice) - 1
            
            if 0 <= sample_idx < len(dataset):
                return sample_idx
            else:
                print(f"Please enter a number between 1 and {len(dataset)}")
                
        except ValueError:
            print("Please enter a valid number")

def select_visualization_backend():
    """Select visualization backend"""
    print("\nVisualization options:")
    print("1. Matplotlib (works on all systems)")
    print("2. Open3D (better 3D interaction, requires installation)")
    
    while True:
        choice = input("Choose visualization backend (1 or 2): ").strip()
        if choice == "1":
            return "matplotlib"
        elif choice == "2":
            return "open3d"
        else:
            print("Please enter 1 or 2")

def select_visualization_type():
    """Select what to visualize"""
    print("\nWhat would you like to visualize?")
    print("1. Point cloud only")
    print("2. Wireframe only") 
    print("3. Both (combined view)")
    
    while True:
        choice = input("Choose visualization type (1, 2, or 3): ").strip()
        if choice in ["1", "2", "3"]:
            return int(choice)
        else:
            print("Please enter 1, 2, or 3")

def main():
    """Main visualization function"""
    print("=" * 50)
    print("3D BUILDING WIREFRAME VISUALIZATION")
    print("=" * 50)
    
    # Load dataset configuration
    dataset_config = cfg_from_yaml_file('../datasets/dataset_config.yaml')
    
    # Fix the root_dir path to be relative to the parent directory
    if dataset_config.Building3D.root_dir.startswith('./'):
        dataset_config.Building3D.root_dir = '../' + dataset_config.Building3D.root_dir[2:]
    
    datasets = build_dataset(dataset_config.Building3D)
    
    # Select dataset split
    print("\nSelect dataset:")
    print("1. Train dataset")
    print("2. Test dataset")
    
    while True:
        split_choice = input("Choose dataset (1 or 2): ").strip()
        if split_choice == "1":
            dataset = datasets['train']
            break
        elif split_choice == "2":
            dataset = datasets['test']
            break
        else:
            print("Please enter 1 or 2")
    
    # Select sample
    sample_idx = select_sample(dataset)
    sample_data = dataset[sample_idx]
    
    # Extract data
    point_cloud = sample_data['point_clouds']
    vertices = sample_data['wf_vertices']
    edges = sample_data['wf_edges']
    
    print(f"\nSample information:")
    print(f"  Point cloud shape: {point_cloud.shape}")
    print(f"  Vertices shape: {vertices.shape}")
    print(f"  Edges shape: {edges.shape}")
    
    # Select visualization type
    viz_type = select_visualization_type()
    
    # Select visualization backend
    backend = select_visualization_backend()
    
    # Perform visualization
    sample_name = os.path.basename(dataset.pc_files[sample_idx]).replace('.xyz', '')
    
    if backend == "matplotlib":
        if viz_type == 1:  # Point cloud only
            fig, ax = visualize_point_cloud_matplotlib(point_cloud, f"Point Cloud - {sample_name}")
            plt.show()
        elif viz_type == 2:  # Wireframe only
            fig, ax = visualize_wireframe_matplotlib(vertices, edges, f"Wireframe - {sample_name}")
            plt.show()
        else:  # Combined
            fig = visualize_combined_matplotlib(point_cloud, vertices, edges, f"Combined View - {sample_name}")
            plt.show()
            
    else:  # Open3D
        if viz_type == 1:  # Point cloud only
            visualize_point_cloud_open3d(point_cloud, f"Point Cloud - {sample_name}")
        elif viz_type == 2:  # Wireframe only
            visualize_wireframe_open3d(vertices, edges, f"Wireframe - {sample_name}")
        else:  # Combined
            visualize_combined_open3d(point_cloud, vertices, edges, f"Combined View - {sample_name}")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
