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

def visualize_point_cloud_matplotlib(point_cloud, title="Point Cloud", max_points=1000, border_weights=None):
    """Visualize point cloud using matplotlib"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points if too many
    if len(point_cloud) > max_points:
        indices = np.random.choice(len(point_cloud), max_points, replace=False)
        sampled_points = point_cloud[indices]
        if border_weights is not None:
            sampled_border_weights = border_weights[indices]
        else:
            sampled_border_weights = None
    else:
        sampled_points = point_cloud
        sampled_border_weights = border_weights
    
    # Extract coordinates
    x, y, z = sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2]
    
    # Choose coloring scheme
    if sampled_border_weights is not None:
        # Color by border weights
        scatter = ax.scatter(x, y, z, c=sampled_border_weights, cmap='hot', s=2, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Border Weights', shrink=0.6)
        title += " (Border Weights)"
    elif sampled_points.shape[1] > 6:  # Has RGB
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

def create_normal_lines(points, normals, normal_length=0.05, surface_labels=None):
    """Create line segments to visualize surface normals with optional surface-type coloring"""
    try:
        import open3d as o3d
        
        # Create line endpoints
        start_points = points
        end_points = points + normals * normal_length
        
        # Combine all points
        all_points = np.vstack([start_points, end_points])
        
        # Create line indices (each normal is one line)
        n_points = len(points)
        lines = np.array([[i, i + n_points] for i in range(n_points)])
        
        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(all_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        # Color normal lines based on surface type if available
        if surface_labels is not None:
            surface_normal_colors = {
                0: [1.0, 0.6, 0.3],  # Floor normals: Orange
                1: [0.3, 1.0, 0.3],  # Wall normals: Bright green
                2: [0.3, 0.3, 1.0],  # Roof normals: Bright blue
                3: [1.0, 1.0, 0.3],  # Edge normals: Bright yellow
                4: [0.7, 0.7, 0.7]   # Unknown normals: Light gray
            }
            colors = []
            for label in surface_labels:
                color = surface_normal_colors.get(int(label), [0.0, 1.0, 1.0])  # Default cyan
                colors.append(color)
        else:
            # Default cyan for all normals
            colors = [[0.0, 1.0, 1.0] for _ in range(len(lines))]  # Cyan
        
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        return line_set
    except ImportError:
        return None

def color_points_by_normal_quality(point_cloud):
    """
    Color points based on their normal orientation quality.
    Green = upward normals (good), Yellow = side normals (ok), Red = downward normals (suspicious)
    """
    try:
        import open3d as o3d
        
        if point_cloud.shape[1] < 6:
            return None  # No normals available
            
        normals = point_cloud[:, 3:6]  # Normals are in columns 3-5, not 8-11!
        z_component = normals[:, 2]
        
        # Create color map based on normal Z component
        colors = np.zeros((len(point_cloud), 3))
        
        # Upward normals (z > 0.5) -> Green
        upward_mask = z_component > 0.5
        colors[upward_mask] = [0.0, 1.0, 0.0]  # Green
        
        # Side/oblique normals (-0.3 <= z <= 0.5) -> Yellow/Orange
        side_mask = (z_component >= -0.3) & (z_component <= 0.5)
        # Interpolate from yellow to orange based on angle
        side_ratio = (z_component[side_mask] + 0.3) / 0.8  # Normalize to [0,1]
        colors[side_mask, 0] = 1.0  # Red component
        colors[side_mask, 1] = 0.5 + 0.5 * side_ratio  # Green component (0.5 to 1.0)
        colors[side_mask, 2] = 0.0  # Blue component
        
        # Downward normals (z < -0.3) -> Red (suspicious outliers)
        downward_mask = z_component < -0.3
        colors[downward_mask] = [1.0, 0.0, 0.0]  # Red
        
        # Print statistics
        n_up = np.sum(upward_mask)
        n_side = np.sum(side_mask)
        n_down = np.sum(downward_mask)
        total = len(point_cloud)
        
        print(f"\n=== Normal Quality Analysis ===")
        print(f"🟢 Upward normals (z > 0.5): {n_up} ({100*n_up/total:.1f}%) - Good quality")
        print(f"🟡 Side normals (-0.3 ≤ z ≤ 0.5): {n_side} ({100*n_side/total:.1f}%) - Acceptable")
        print(f"🔴 Downward normals (z < -0.3): {n_down} ({100*n_down/total:.1f}%) - Check for outliers")
        
        return colors
        
    except ImportError:
        return None


def color_points_by_surface_type(point_cloud):
    """
    Color points based on their surface type labels (if available).
    Falls back to normal quality coloring if no surface labels.
    """
    try:
        import open3d as o3d
        
        # Since we removed surface type classification, fallback to normal-based coloring
        return color_points_by_normal_quality(point_cloud)
        
    except ImportError:
        return None


def color_points_by_group_instance(point_cloud):
    """
    Color points based on their group instance IDs (if available).
    Each unique group/instance gets its own distinct color for better visualization.
    """
    try:
        import open3d as o3d
        
        if point_cloud.shape[1] < 7:
            # No group IDs, fallback to normal coloring
            return color_points_by_normal_quality(point_cloud)
            
        group_ids = point_cloud[:, 6].astype(int)
        unique_groups = np.unique(group_ids)
        n_groups = len(unique_groups)
        
        if n_groups == 0:
            return color_points_by_normal_quality(point_cloud)
        
        # Generate distinct colors using HSV color space for better separation
        colors = np.zeros((len(point_cloud), 3))
        
        # Use a color palette that provides good visual separation
        from matplotlib import cm
        import matplotlib.colors as mcolors
        
        # Generate colors using a colormap with good separation
        if n_groups <= 10:
            # Use tab10 for small number of groups (best distinction)
            colormap = cm.get_cmap('tab10')
        elif n_groups <= 20:
            # Use tab20 for medium number of groups
            colormap = cm.get_cmap('tab20')
        else:
            # Use hsv for large number of groups
            colormap = cm.get_cmap('hsv')
        
        # Assign colors to each group
        group_colors = {}
        
        print(f"\n=== Group Instance Distribution ===")
        total = len(point_cloud)
        
        for i, group_id in enumerate(unique_groups):
            # Get color from colormap
            color_normalized = i / max(n_groups - 1, 1)  # Normalize to [0,1]
            rgb_color = colormap(color_normalized)[:3]  # Extract RGB, ignore alpha
            group_colors[group_id] = rgb_color
            
            # Assign color to all points in this group
            mask = group_ids == group_id
            colors[mask] = rgb_color
            
            # Print group statistics
            count = np.sum(mask)
            color_rgb = [int(c*255) for c in rgb_color]
            print(f"🔶 Group {group_id}: {count} points ({100*count/total:.1f}%) - RGB{color_rgb}")
        
        return colors
        
    except ImportError:
        return None


def color_points_by_border_weights(point_cloud, border_weights=None):
    """
    Color points based on their border weights (geometric border-ness).
    Highlights edges and corners that form object borders.
    
    Args:
        point_cloud (np.ndarray): Point cloud data
        border_weights (np.ndarray, optional): Pre-computed border weights. If None, will compute them.
    """
    try:
        import open3d as o3d
        
        # Extract only XYZ coordinates for weight calculation
        points_xyz = point_cloud[:, :3]
        
        # Use provided border weights or compute them
        if border_weights is None:
            border_weights = compute_border_weights(points_xyz, k_neighbors=25, normalize_weights=True)
        else:
            # Ensure border_weights is the right length
            if len(border_weights) != len(points_xyz):
                print(f"Warning: border_weights length {len(border_weights)} doesn't match points {len(points_xyz)}")
                border_weights = compute_border_weights(points_xyz, k_neighbors=25, normalize_weights=True)
        
        # Create color map: blue (low border weight) to red (high border weight)
        colors = np.zeros((len(point_cloud), 3))
        
        # Use a blue-to-red colormap for border weights
        from matplotlib import cm
        colormap = cm.get_cmap('coolwarm')  # Blue-to-red
        
        # Apply colormap
        for i, weight in enumerate(border_weights):
            rgb_color = colormap(weight)[:3]  # Extract RGB, ignore alpha
            colors[i] = rgb_color
        
        print(f"\n=== Border Weight Distribution ===")
        high_border = np.sum(border_weights > 0.7)
        medium_border = np.sum((border_weights > 0.3) & (border_weights <= 0.7))
        low_border = np.sum(border_weights <= 0.3)
        total = len(point_cloud)
        
        print(f"🔵 Low border weights (≤0.3): {low_border} points ({100*low_border/total:.1f}%) - Blue")
        print(f"🟢 Medium border weights (0.3-0.7): {medium_border} points ({100*medium_border/total:.1f}%) - Green")
        print(f"🔴 High border weights (>0.7): {high_border} points ({100*high_border/total:.1f}%) - Red")
        
        return colors
        
    except ImportError:
        return None





def visualize_point_cloud_open3d(point_cloud, title="Point Cloud", show_coordinate_frame=True, show_boundaries=False, show_normals=False, color_by_normal_quality=False, color_by_surface_type=False, color_by_group_instance=False, color_by_border_weights=False, coordinate_frame_size=0.1, border_weights=None):
    """Visualize point cloud using Open3D with simple axis lines, optional boundaries, and explicit normal lines"""
    try:
        import open3d as o3d
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Choose coloring scheme (priority order)
        if color_by_border_weights:
            # Border weight colors (highest priority for weight analysis)
            border_colors = color_points_by_border_weights(point_cloud, border_weights)
            if border_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(border_colors)
            else:
                print("⚠️  Border weight calculation failed, using group instance colors")
                color_by_group_instance = True
        elif color_by_group_instance:
            # Color by group instance (second priority)
            group_colors = color_points_by_group_instance(point_cloud)
            if group_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(group_colors)
            else:
                print("⚠️  No group instance data available, using surface type colors")
                color_by_surface_type = True
        elif color_by_surface_type:
            # Color by surface type (third priority)
            surface_colors = color_points_by_surface_type(point_cloud)
            if surface_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(surface_colors)
            else:
                print("⚠️  No surface labels available, using default colors")
        elif color_by_normal_quality:
            # Color by normal quality
            quality_colors = color_points_by_normal_quality(point_cloud)
            if quality_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(quality_colors)
            else:
                print("⚠️  No normals available for quality coloring, using default colors")
        else:
            # Add regular colors if available
            if point_cloud.shape[1] > 6:
                colors = point_cloud[:, 3:6]
                if colors.max() > 1.0:  # Normalize if needed
                    colors = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create list of geometries to visualize
        geometries = [pcd]
        
        # Add explicit normal visualization if available and requested
        has_normals = point_cloud.shape[1] >= 6  # Need at least 6 columns for XYZ + normals
        if has_normals and show_normals:
            normals = point_cloud[:, 3:6]  # Normals are in columns 3-5, not 8-11!
            points = point_cloud[:, :3]
            
            # Set point cloud normals for Open3D
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Check if surface labels are available for normal coloring
            surface_labels = None
            # With the simplified structure, we don't have surface labels anymore
            # Normal arrows will be colored uniformly (cyan)
            
            # Create explicit normal line segments for better visibility
            normal_lines = create_normal_lines(points, normals, normal_length=0.03, surface_labels=surface_labels)
            if normal_lines is not None:
                geometries.append(normal_lines)
                print(f"✅ Visualizing {len(normals)} surface normals as cyan line segments")
            else:
                print(f"✅ Visualizing {len(normals)} surface normals (Open3D built-in)")
        elif has_normals:
            print(f"ℹ️  Point cloud has normals (use show_normals=True to display)")
        
        # Add geometries list continues...
        
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

def visualize_combined_open3d(point_cloud, vertices, edges, title="Combined View", show_coordinate_frame=True, show_boundaries=False, show_normals=False, color_by_normal_quality=False, color_by_surface_type=False, color_by_group_instance=False, color_by_border_weights=False, coordinate_frame_size=0.1, border_weights=None):
    """Visualize point cloud and wireframe together using Open3D with coordinate frame, boundaries, and optional normals"""
    try:
        import open3d as o3d
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        
        # Choose coloring scheme for point cloud (priority order)
        if color_by_border_weights:
            # Border weight colors (highest priority for weight analysis)
            border_colors = color_points_by_border_weights(point_cloud, border_weights)
            if border_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(border_colors)
            else:
                print("⚠️  Border weight calculation failed, using group instance colors")
                color_by_group_instance = True
        elif color_by_group_instance:
            # Color by group instance (second priority)
            group_colors = color_points_by_group_instance(point_cloud)
            if group_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(group_colors)
            else:
                print("⚠️  No group instance data available, using surface type colors")
                surface_colors = color_points_by_surface_type(point_cloud)
                if surface_colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(surface_colors)
                else:
                    print("⚠️  No surface labels available, using default colors")
        elif color_by_surface_type:
            # Color by surface type (third priority)
            surface_colors = color_points_by_surface_type(point_cloud)
            if surface_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(surface_colors)
            else:
                print("⚠️  No surface labels available, using default colors")
        elif color_by_normal_quality:
            # Color by normal quality
            quality_colors = color_points_by_normal_quality(point_cloud)
            if quality_colors is not None:
                pcd.colors = o3d.utility.Vector3dVector(quality_colors)
            else:
                print("⚠️  No normals available for quality coloring, using default colors")
        else:
            # Use regular colors
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
        
        # Add explicit normal visualization if available and requested
        has_normals = point_cloud.shape[1] >= 6  # Need at least 6 columns for XYZ + normals
        if has_normals and show_normals:
            normals = point_cloud[:, 3:6]  # Normals are in columns 3-5, not 8-11!
            points = point_cloud[:, :3]
            
            # Set point cloud normals for Open3D
            pcd.normals = o3d.utility.Vector3dVector(normals)
            
            # Check if surface labels are available for normal coloring
            surface_labels = None
            # With the simplified structure, we don't have surface labels anymore
            # Normal arrows will be colored uniformly (cyan)
            
            # Create explicit normal line segments for better visibility
            normal_lines = create_normal_lines(points, normals, normal_length=0.03, surface_labels=surface_labels)
            if normal_lines is not None:
                geometries.append(normal_lines)
                print(f"✅ Visualizing {len(normals)} surface normals as cyan line segments on point cloud")
            else:
                print(f"✅ Visualizing {len(normals)} surface normals on point cloud (Open3D built-in)")
        elif has_normals:
            print(f"ℹ️  Point cloud has normals (use show_normals=True to display)")
        
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

def select_outlier_removal():
    """Select whether to enable outlier removal in the pipeline"""
    print("\nOutlier removal options:")
    print("1. Enable outlier removal (recommended for cleaner visualization)")
    print("2. Disable outlier removal (show all raw points)")
    
    while True:
        choice = input("Choose outlier removal option (1 or 2): ").strip()
        if choice == "1":
            return True
        elif choice == "2":
            return False
        else:
            print("Please enter 1 or 2")

def select_coordinate_options():
    """Select coordinate visualization options for Open3D"""
    print("\nCoordinate frame options:")
    print("1. Show coordinate frame at origin")
    print("2. Show coordinate frame + boundaries")
    print("3. Show coordinate frame + boundaries + normals")
    print("4. Show coordinate frame + boundaries + normal quality colors")
    print("5. Show coordinate frame + boundaries + group instance colors")
    print("6. Show coordinate frame + boundaries + group instance colors + normals")
    print("7. Show coordinate frame + boundaries + border weight colors")
    print("8. No coordinate aids")
    
    while True:
        choice = input("Choose coordinate option (1-8): ").strip()
        if choice == "1":
            return True, False, False, False, False, False, False  # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "2":
            return True, True, False, False, False, False, False   # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "3":
            return True, True, True, False, False, False, False    # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "4":
            return True, True, False, True, False, False, False    # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "5":
            return True, True, False, False, False, True, False    # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "6":
            return True, True, True, False, False, True, False     # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "7":
            return True, True, False, False, False, False, True    # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        elif choice == "8":
            return False, False, False, False, False, False, False # show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights
        else:
            print("Please enter 1-8")

def main():
    """Main visualization function"""
    print("=" * 50)
    print("3D BUILDING WIREFRAME VISUALIZATION")
    print("=" * 50)
    
    # Select outlier removal option
    use_outlier_removal = select_outlier_removal()
    
    # Load dataset configuration
    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    
    # Update outlier removal setting based on user choice
    dataset_config.Building3D.use_outlier_removal = use_outlier_removal
    print(f"Outlier removal: {'Enabled' if use_outlier_removal else 'Disabled'}")
    
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
    

    # Using dataset preprocessing - extract border weights if available
    border_weights = None
    if point_cloud.shape[1] > 3:  # Check if border weights are already computed
        # Assume last column contains border weights (from dataset preprocessing)
        potential_weights = point_cloud[:, -1]
        if np.all((potential_weights >= 0) & (potential_weights <= 1)):
            border_weights = potential_weights
            print(f"\nFound pre-computed border weights in dataset:")
            print(f"  Border weights range: [{border_weights.min():.4f}, {border_weights.max():.4f}]")
            print(f"  Border weights mean: {border_weights.mean():.4f}")
        
    if border_weights is None:
        # Compute border weights for visualization
        print("\nComputing border weights for visualization...")
        border_weights = compute_border_weights(point_cloud[:, :3], k_neighbors=15)
        print(f"  Border weights computed: range [{border_weights.min():.4f}, {border_weights.max():.4f}]")
    
    # Select visualization type
    viz_type = select_visualization_type()
    
    # Select visualization backend
    backend = select_visualization_backend()
    
    # Get coordinate options for Open3D
    show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights = False, False, False, False, False, False, False
    if backend == "open3d":
        show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights = select_coordinate_options()
    
    # Perform visualization
    sample_name = os.path.basename(dataset.pc_files[sample_idx]).replace('.xyz', '')
    
    if backend == "matplotlib":
        if viz_type == 1:  # Point cloud only
            fig, ax = visualize_point_cloud_matplotlib(point_cloud, f"Point Cloud - {sample_name}", border_weights=border_weights)
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
                show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights, coord_size, border_weights
            )
        elif viz_type == 2:  # Wireframe only
            visualize_wireframe_open3d(
                vertices, edges, f"Wireframe - {sample_name}",
                show_frame, show_boundaries, coord_size
            )
        else:  # Combined
            visualize_combined_open3d(
                point_cloud, vertices, edges, f"Combined View - {sample_name}",
                show_frame, show_boundaries, show_normals, color_by_normal_quality, color_by_surface_type, color_by_group_instance, color_by_border_weights, coord_size, border_weights
            )
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
