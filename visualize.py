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

def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg

def create_coordinate_frames(positions, size=0.1):
    """Create coordinate frames at multiple positions"""
    try:
        import open3d as o3d
        frames = []
        for pos in positions:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size, origin=pos
            )
            frames.append(frame)
        return frames
    except ImportError:
        return []

def add_boundary_box(min_coords, max_coords):
    """Create boundary box lines for better spatial reference"""
    try:
        import open3d as o3d
        
        # Define the 8 corners of the bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],  # 0: min corner
            [max_coords[0], min_coords[1], min_coords[2]],  # 1: +X
            [min_coords[0], max_coords[1], min_coords[2]],  # 2: +Y
            [max_coords[0], max_coords[1], min_coords[2]],  # 3: +X+Y
            [min_coords[0], min_coords[1], max_coords[2]],  # 4: +Z
            [max_coords[0], min_coords[1], max_coords[2]],  # 5: +X+Z
            [min_coords[0], max_coords[1], max_coords[2]],  # 6: +Y+Z
            [max_coords[0], max_coords[1], max_coords[2]]   # 7: max corner
        ])
        
        # Define the 12 edges of the bounding box
        edges = np.array([
            # Bottom face (Z = min)
            [0, 1], [1, 3], [3, 2], [2, 0],
            # Top face (Z = max)
            [4, 5], [5, 7], [7, 6], [6, 4],
            # Vertical edges
            [0, 4], [1, 5], [2, 6], [3, 7]
        ])
        
        # Create line set for boundary box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        
        # Color boundary lines yellow/orange for visibility
        colors = [[1.0, 0.6, 0.0] for _ in range(len(edges))]  # Orange
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    except ImportError:
        return None

def visualize_with_coordinate_display(geometries, title, point_cloud=None):
    """Enhanced visualization with coordinate frame and better interaction"""
    try:
        import open3d as o3d
        
        # Create visualizer with better settings
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1200, height=800)
        
        # Add geometries
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Configure render options for better visualization
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.line_width = 2.0
        
        # Get view control for better default view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        # Run the visualizer
        vis.run()
        vis.destroy_window()
        
    except ImportError:
        pass
    except Exception as e:
        # Fallback to basic visualization
        try:
            import open3d as o3d
            o3d.visualization.draw_geometries(geometries, window_name=title)
        except:
            pass

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

def visualize_point_cloud_open3d(point_cloud, title="Point Cloud", show_coordinate_frame=True, show_boundaries=False, coordinate_frame_size=0.1):
    """Visualize point cloud using Open3D with simple axis lines and optional boundaries"""
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
        
        # Create list of geometries to visualize
        geometries = [pcd]
        
        # Add simple axis lines instead of arrows
        if show_coordinate_frame:
            axis_size = coordinate_frame_size
            
            # Create simple line segments for X, Y, Z axes
            # X axis (red)
            x_line = o3d.geometry.LineSet()
            x_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [axis_size, 0, 0]])
            x_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            x_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
            geometries.append(x_line)
            
            # Y axis (green)
            y_line = o3d.geometry.LineSet()
            y_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, axis_size, 0]])
            y_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            y_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
            geometries.append(y_line)
            
            # Z axis (blue)
            z_line = o3d.geometry.LineSet()
            z_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, axis_size]])
            z_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            z_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue
            geometries.append(z_line)
        
        # Add boundary box if requested
        if show_boundaries:
            coords = point_cloud[:, :3]
            min_coords = np.min(coords, axis=0)
            max_coords = np.max(coords, axis=0)
            boundary_box = add_boundary_box(min_coords, max_coords)
            if boundary_box is not None:
                geometries.append(boundary_box)
                
            # Print boundary information
            print(f"\n=== Data Boundaries for {title} ===")
            print(f"X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
            print(f"Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
            print(f"Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
        
        # Simple visualization without extra coordinate display
        o3d.visualization.draw_geometries(geometries, window_name=title)
        
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

def visualize_wireframe_open3d(vertices, edges, title="Wireframe", show_coordinate_frame=True, show_boundaries=False, coordinate_frame_size=0.1):
    """Visualize wireframe using Open3D with coordinate text labels on vertices and optional boundaries"""
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
        
        # Create list of geometries to visualize
        geometries = [line_set, pcd]
        
        # Add simple axis lines instead of arrows
        if show_coordinate_frame:
            axis_size = coordinate_frame_size
            
            # Create simple line segments for X, Y, Z axes
            # X axis (red)
            x_line = o3d.geometry.LineSet()
            x_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [axis_size, 0, 0]])
            x_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            x_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
            geometries.append(x_line)
            
            # Y axis (green)
            y_line = o3d.geometry.LineSet()
            y_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, axis_size, 0]])
            y_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            y_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
            geometries.append(y_line)
            
            # Z axis (blue)
            z_line = o3d.geometry.LineSet()
            z_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, axis_size]])
            z_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            z_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue
            geometries.append(z_line)
        
        # Add boundary box if requested
        if show_boundaries:
            min_coords = np.min(vertices, axis=0)
            max_coords = np.max(vertices, axis=0)
            boundary_box = add_boundary_box(min_coords, max_coords)
            if boundary_box is not None:
                geometries.append(boundary_box)
                
            # Print boundary information
            print(f"\n=== Wireframe Boundaries for {title} ===")
            print(f"X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
            print(f"Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
            print(f"Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
        
        # Create text labels for each vertex showing coordinates
        # Note: Open3D doesn't have built-in text rendering, so we'll print coordinates to console for now
        print(f"\n=== Vertex Coordinates for {title} ===")
        for i, vertex in enumerate(vertices):
            print(f"Vertex {i}: ({vertex[0]:.3f}, {vertex[1]:.3f}, {vertex[2]:.3f})")
        
        # Simple visualization
        o3d.visualization.draw_geometries(geometries, window_name=title)
        
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

def visualize_combined_open3d(point_cloud, vertices, edges, title="Combined View", show_coordinate_frame=True, show_boundaries=False, coordinate_frame_size=0.1):
    """Visualize point cloud and wireframe together using Open3D with coordinate frame and boundaries"""
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
        
        # Create list of geometries to visualize
        geometries = [pcd, line_set, vertex_pcd]
        
        # Add simple axis lines instead of arrows
        if show_coordinate_frame:
            axis_size = coordinate_frame_size
            
            # Create simple line segments for X, Y, Z axes
            # X axis (red)
            x_line = o3d.geometry.LineSet()
            x_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [axis_size, 0, 0]])
            x_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            x_line.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # Red
            geometries.append(x_line)
            
            # Y axis (green)
            y_line = o3d.geometry.LineSet()
            y_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, axis_size, 0]])
            y_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            y_line.colors = o3d.utility.Vector3dVector([[0, 1, 0]])  # Green
            geometries.append(y_line)
            
            # Z axis (blue)
            z_line = o3d.geometry.LineSet()
            z_line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, axis_size]])
            z_line.lines = o3d.utility.Vector2iVector([[0, 1]])
            z_line.colors = o3d.utility.Vector3dVector([[0, 0, 1]])  # Blue
            geometries.append(z_line)
        
        # Add boundary box if requested
        if show_boundaries:
            # Combine point cloud and wireframe coordinates for overall boundaries
            all_coords = np.vstack([point_cloud[:, :3], vertices])
            min_coords = np.min(all_coords, axis=0)
            max_coords = np.max(all_coords, axis=0)
            boundary_box = add_boundary_box(min_coords, max_coords)
            if boundary_box is not None:
                geometries.append(boundary_box)
                
            # Print boundary information
            print(f"\n=== Combined Data Boundaries for {title} ===")
            print(f"X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
            print(f"Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
            print(f"Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
        
        # Simple visualization
        o3d.visualization.draw_geometries(geometries, window_name=title)
        
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

def select_coordinate_options():
    """Select coordinate visualization options for Open3D"""
    print("\nCoordinate frame options:")
    print("1. Show coordinate frame at origin")
    print("2. Show coordinate frame + boundaries")
    print("3. No coordinate aids")
    
    while True:
        choice = input("Choose coordinate option (1, 2, or 3): ").strip()
        if choice == "1":
            return True, False  # show_frame, show_boundaries
        elif choice == "2":
            return True, True   # show_frame, show_boundaries
        elif choice == "3":
            return False, False # show_frame, show_boundaries
        else:
            print("Please enter 1, 2, or 3")

def main():
    """Main visualization function"""
    print("=" * 50)
    print("3D BUILDING WIREFRAME VISUALIZATION")
    print("=" * 50)
    
    # Load dataset configuration
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
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
    
    # Get coordinate options for Open3D
    show_frame, show_boundaries = False, False
    if backend == "open3d":
        show_frame, show_boundaries = select_coordinate_options()
    
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
        # Determine coordinate frame size based on data bounds
        coord_size = 0.1
        if show_frame:
            bounds = np.max(np.abs(point_cloud[:, :3]))
            coord_size = bounds * 0.1  # 10% of data bounds
            
        if viz_type == 1:  # Point cloud only
            visualize_point_cloud_open3d(
                point_cloud, f"Point Cloud - {sample_name}", 
                show_frame, show_boundaries, coord_size
            )
        elif viz_type == 2:  # Wireframe only
            visualize_wireframe_open3d(
                vertices, edges, f"Wireframe - {sample_name}",
                show_frame, show_boundaries, coord_size
            )
        else:  # Combined
            visualize_combined_open3d(
                point_cloud, vertices, edges, f"Combined View - {sample_name}",
                show_frame, show_boundaries, coord_size
            )
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
