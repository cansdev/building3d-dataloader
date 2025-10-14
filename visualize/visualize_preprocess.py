#!/usr/bin/env python3
"""
Preprocessing Visualization Script

This script visualizes the preprocessing steps applied to Building3D dataset:
1. Original point cloud
2. After normalization (centered and scaled)
3. After spatial grouping (different colors for each group)
4. After outlier removal (removed points shown in red)
5. Edge weights visualization (heatmap)
6. Cross weights visualization (heatmap)
7. Border weights visualization (combined edge + cross)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset
from utils.surface_grouping import simple_spatial_grouping
from utils.outlier_removal import remove_statistical_outliers
from utils.weight_assignment import calculate_edge_weights, calculate_edgecross_weights, compute_border_weights, calculate_local_features
import yaml
from easydict import EasyDict

# Create custom aggressive colormap: black → red → bright yellow
# 0.0 = black, 0.2 = dark red, 0.5 = bright red, 1.0 = bright yellow
black_red_yellow = LinearSegmentedColormap.from_list('black_red_yellow', [
    (0.0, (0.0, 0.0, 0.0)),      # black at 0.0
    (0.2, (0.5, 0.0, 0.0)),      # dark red at 0.2
    (0.4, (1.0, 0.0, 0.0)),      # bright red at 0.4
    (0.6, (1.0, 0.3, 0.0)),      # orange-red at 0.6
    (0.8, (1.0, 0.7, 0.0)),      # orange-yellow at 0.8
    (1.0, (1.0, 1.0, 0.0))       # bright yellow at 1.0
])


def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def load_point_cloud(dataset, sample_idx):
    """Load raw point cloud from dataset"""
    pc_file = dataset.pc_files[sample_idx]
    pc = np.loadtxt(pc_file)
    return pc


def normalize_point_cloud(pc):
    """Step 1: Normalize point cloud (centroid + scale)"""
    xyz = pc[:, :3]
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
    
    if max_distance < 1e-6:
        max_distance = 1.0
        
    xyz_normalized = xyz_centered / max_distance
    
    # Rebuild point cloud with normalized coords
    normalized_pc = pc.copy()
    normalized_pc[:, :3] = xyz_normalized
    
    # Normalize colors if present (RGBA: columns 3-6)
    if normalized_pc.shape[1] >= 7:
        normalized_pc[:, 3:7] = normalized_pc[:, 3:7] / 256.0
        
    return centroid, max_distance, normalized_pc


def apply_spatial_grouping(points, config):
    """Step 2: Spatial grouping"""
    params = config.Building3D.preprocessor.grouping_params.coarse_params
    
    surface_groups = simple_spatial_grouping(
        points,
        eps=params.eps,
        min_samples=params.min_samples,
        auto_scale_eps=params.auto_scale_eps
    )
    
    # Create group_id array
    group_ids = np.zeros(len(points), dtype=np.int32)
    for i, group in enumerate(surface_groups):
        group_ids[group.indices] = i
        
    return group_ids, surface_groups


def apply_outlier_removal(pc, group_ids, config):
    """Step 3: Outlier removal per group"""
    params = config.Building3D.preprocessor.outlier_params
    points = pc[:, :3]
    
    # Apply outlier removal per group
    keep_mask = np.ones(len(points), dtype=bool)
    
    for group_id in np.unique(group_ids):
        group_mask = group_ids == group_id
        group_points = points[group_mask]
        
        if len(group_points) < 10:
            continue
            
        # Statistical outlier removal
        try:
            _, inlier_mask = remove_statistical_outliers(
                group_points,
                k_neighbors=params.stat_k,
                std_ratio=params.stat_std_ratio
            )
            
            # Update global mask
            group_indices = np.where(group_mask)[0]
            keep_mask[group_indices[~inlier_mask]] = False
        except Exception as e:
            print(f"Warning: Outlier removal failed for group {group_id}: {e}")
            continue
    
    return keep_mask


def apply_weight_computation(points, config):
    """Step 4: Compute border weights"""
    params = config.Building3D.preprocessor.weight_params
    
    # Compute edge weights
    eigenvalues = calculate_local_features(points, params.k_neighbors)
    edge_weights = calculate_edge_weights(eigenvalues)
    
    # Compute cross weights
    vertex_weights = calculate_edgecross_weights(eigenvalues)
    
    # Compute border weights (combined)
    border_weights = compute_border_weights(
        points,
        k_neighbors=params.k_neighbors,
        normalize_weights=params.normalize_weights,
        multi_scale=params.multi_scale
    )
    
    return edge_weights, vertex_weights, border_weights


def visualize_preprocessing_steps(pc, config, sample_name):
    """Visualize all preprocessing steps"""
    
    print(f"\nProcessing sample: {sample_name}")
    print(f"Original point cloud shape: {pc.shape}")
    
    # Step 1: Normalization
    print("\n[1/4] Normalizing point cloud...")
    centroid, max_distance, normalized_pc = normalize_point_cloud(pc)
    print(f"  Centroid: {centroid}")
    print(f"  Max distance: {max_distance:.4f}")
    
    # Step 2: Spatial Grouping
    print("\n[2/4] Applying spatial grouping...")
    group_ids, surface_groups = apply_spatial_grouping(normalized_pc[:, :3], config)
    n_groups = len(np.unique(group_ids))
    print(f"  Number of groups: {n_groups}")
    for i, group in enumerate(surface_groups[:5]):  # Show first 5 groups
        print(f"    Group {i}: {group.size} points")
    
    # Step 3: Outlier Removal
    print("\n[3/4] Removing outliers...")
    keep_mask = apply_outlier_removal(normalized_pc, group_ids, config)
    n_outliers = (~keep_mask).sum()
    print(f"  Removed {n_outliers} outliers ({n_outliers/len(pc)*100:.2f}%)")
    
    # Apply mask
    filtered_pc = normalized_pc[keep_mask]
    filtered_group_ids = group_ids[keep_mask]
    
    # Step 4: Weight Computation
    print("\n[4/4] Computing geometric weights...")
    edge_weights, vertex_weights, border_weights = apply_weight_computation(
        filtered_pc[:, :3], config
    )
    print(f"  Edge weights range: [{edge_weights.min():.3f}, {edge_weights.max():.3f}]")
    print(f"  Cross weights range: [{vertex_weights.min():.3f}, {vertex_weights.max():.3f}]")
    print(f"  Border weights range: [{border_weights.min():.3f}, {border_weights.max():.3f}]")
    print(f"  Points with high edge weights (>0.5): {(edge_weights > 0.5).sum()}")
    print(f"  Points with high cross weights (>0.5): {(vertex_weights > 0.5).sum()}")
    print(f"  Points with high border weights (>0.5): {(border_weights > 0.5).sum()}")
    
    # Subsample for visualization if too many points
    max_points = 5000
    if len(pc) > max_points:
        viz_indices = np.random.choice(len(pc), max_points, replace=False)
        viz_pc_orig = pc[viz_indices]
        viz_pc_norm = normalized_pc[viz_indices]
        viz_group_ids = group_ids[viz_indices]
        viz_keep_mask = keep_mask[viz_indices]
    else:
        viz_pc_orig = pc
        viz_pc_norm = normalized_pc
        viz_group_ids = group_ids
        viz_keep_mask = keep_mask
    
    if len(filtered_pc) > max_points:
        viz_indices_filtered = np.random.choice(len(filtered_pc), max_points, replace=False)
        viz_filtered_pc = filtered_pc[viz_indices_filtered]
        viz_filtered_group_ids = filtered_group_ids[viz_indices_filtered]
        viz_edge_weights = edge_weights[viz_indices_filtered]
        viz_vertex_weights = vertex_weights[viz_indices_filtered]
        viz_border_weights = border_weights[viz_indices_filtered]
    else:
        viz_filtered_pc = filtered_pc
        viz_filtered_group_ids = filtered_group_ids
        viz_edge_weights = edge_weights
        viz_vertex_weights = vertex_weights
        viz_border_weights = border_weights
    
    # ========================================================================
    # Show individual plots one by one
    # ========================================================================
    
    print("\n" + "="*60)
    print("Showing individual visualizations (close each to see next)...")
    print("="*60)
    
    # Plot 1: Original point cloud
    print("\n[1/8] Original Point Cloud")
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(viz_pc_orig[:, 0], viz_pc_orig[:, 1], viz_pc_orig[:, 2],
                c='lightblue', s=1, alpha=0.6)
    ax1.set_title(f'1. Original Point Cloud - {sample_name}\n({len(pc)} points)', fontsize=14)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
    
    # Plot 2: After normalization
    print("\n[2/8] After Normalization")
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(viz_pc_norm[:, 0], viz_pc_norm[:, 1], viz_pc_norm[:, 2],
                c='lightgreen', s=1, alpha=0.6)
    ax2.set_title(f'2. After Normalization - {sample_name}\n(centered & scaled)', fontsize=14)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.tight_layout()
    plt.show()
    
    # Plot 3: After spatial grouping
    print("\n[3/8] Spatial Grouping")
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111, projection='3d')
    scatter3 = ax3.scatter(viz_pc_norm[:, 0], viz_pc_norm[:, 1], viz_pc_norm[:, 2],
                           c=viz_group_ids, cmap='tab20', s=2, alpha=0.7, vmin=0, vmax=n_groups-1)
    ax3.set_title(f'3. Spatial Grouping - {sample_name}\n({n_groups} groups)', fontsize=14)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    plt.colorbar(scatter3, ax=ax3, shrink=0.5, label='Group ID', ticks=range(n_groups))
    plt.tight_layout()
    plt.show()
    
    # Plot 4: Outlier removal comparison
    print("\n[4/8] Outlier Removal")
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111, projection='3d')
    # Show kept points in green, removed in red
    kept_points = viz_pc_norm[viz_keep_mask]
    removed_points = viz_pc_norm[~viz_keep_mask]
    if len(kept_points) > 0:
        ax4.scatter(kept_points[:, 0], kept_points[:, 1], kept_points[:, 2],
                    c='green', s=1, alpha=0.5, label='Kept')
    if len(removed_points) > 0:
        ax4.scatter(removed_points[:, 0], removed_points[:, 1], removed_points[:, 2],
                    c='red', s=10, marker='x', alpha=0.8, label='Removed')
    ax4.set_title(f'4. Outlier Removal - {sample_name}\n({n_outliers} outliers removed)', fontsize=14)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.legend()
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Edge weights
    print("\n[5/8] Edge Weights")
    fig5 = plt.figure(figsize=(10, 8))
    ax5 = fig5.add_subplot(111, projection='3d')
    scatter5 = ax5.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                           c=viz_edge_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax5.set_title(f'5. Edge Weights - {sample_name}\n({(edge_weights > 0.5).sum()} high-weight points)', fontsize=14)
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    plt.colorbar(scatter5, ax=ax5, shrink=0.5, label='Edge Weight')
    plt.tight_layout()
    plt.show()
    
    # Plot 6: Vertex weights
    print("\n[6/8] Cross Weights")
    fig6 = plt.figure(figsize=(10, 8))
    ax6 = fig6.add_subplot(111, projection='3d')
    scatter6 = ax6.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                           c=viz_vertex_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax6.set_title(f'6. Cross Weights - {sample_name}\n({(vertex_weights > 0.5).sum()} high-weight points)', fontsize=14)
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')
    plt.colorbar(scatter6, ax=ax6, shrink=0.5, label='Cross Weight')
    plt.tight_layout()
    plt.show()
    
    # Plot 7: Border weights (combined)
    print("\n[7/8] Border Weights (Combined)")
    fig7 = plt.figure(figsize=(10, 8))
    ax7 = fig7.add_subplot(111, projection='3d')
    scatter7 = ax7.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                           c=viz_border_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax7.set_title(f'7. Border Weights (Combined) - {sample_name}\n({(border_weights > 0.5).sum()} high-weight points)', fontsize=14)
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    plt.colorbar(scatter7, ax=ax7, shrink=0.5, label='Border Weight')
    plt.tight_layout()
    plt.show()
    
    # Plot 8: Final processed point cloud with group colors
    print("\n[8/8] Final Result")
    fig8 = plt.figure(figsize=(10, 8))
    ax8 = fig8.add_subplot(111, projection='3d')
    n_groups_filtered = len(np.unique(viz_filtered_group_ids))
    scatter8 = ax8.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                           c=viz_filtered_group_ids, cmap='tab20', s=2, alpha=0.7, vmin=0, vmax=n_groups-1)
    ax8.set_title(f'8. Final Result - {sample_name}\n({len(filtered_pc)} points after preprocessing)', fontsize=14)
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_zlabel('Z')
    plt.colorbar(scatter8, ax=ax8, shrink=0.5, label='Group ID', ticks=range(n_groups))
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # Show all plots together in one figure
    # ========================================================================
    
    print("\n" + "="*60)
    print("Showing all visualizations together...")
    print("="*60)
    
    fig_all = plt.figure(figsize=(20, 12))
    
    # Plot 1: Original point cloud
    ax1_all = fig_all.add_subplot(2, 4, 1, projection='3d')
    ax1_all.scatter(viz_pc_orig[:, 0], viz_pc_orig[:, 1], viz_pc_orig[:, 2],
                    c='lightblue', s=1, alpha=0.6)
    ax1_all.set_title(f'1. Original Point Cloud\n({len(pc)} points)')
    ax1_all.set_xlabel('X')
    ax1_all.set_ylabel('Y')
    ax1_all.set_zlabel('Z')
    
    # Plot 2: After normalization
    ax2_all = fig_all.add_subplot(2, 4, 2, projection='3d')
    ax2_all.scatter(viz_pc_norm[:, 0], viz_pc_norm[:, 1], viz_pc_norm[:, 2],
                    c='lightgreen', s=1, alpha=0.6)
    ax2_all.set_title(f'2. After Normalization\n(centered & scaled)')
    ax2_all.set_xlabel('X')
    ax2_all.set_ylabel('Y')
    ax2_all.set_zlabel('Z')
    
    # Plot 3: After spatial grouping
    ax3_all = fig_all.add_subplot(2, 4, 3, projection='3d')
    scatter3_all = ax3_all.scatter(viz_pc_norm[:, 0], viz_pc_norm[:, 1], viz_pc_norm[:, 2],
                                   c=viz_group_ids, cmap='tab20', s=2, alpha=0.7, vmin=0, vmax=n_groups-1)
    ax3_all.set_title(f'3. Spatial Grouping\n({n_groups} groups)')
    ax3_all.set_xlabel('X')
    ax3_all.set_ylabel('Y')
    ax3_all.set_zlabel('Z')
    plt.colorbar(scatter3_all, ax=ax3_all, shrink=0.5, label='Group ID', ticks=range(n_groups))
    
    # Plot 4: Outlier removal comparison
    ax4_all = fig_all.add_subplot(2, 4, 4, projection='3d')
    if len(kept_points) > 0:
        ax4_all.scatter(kept_points[:, 0], kept_points[:, 1], kept_points[:, 2],
                        c='green', s=1, alpha=0.5, label='Kept')
    if len(removed_points) > 0:
        ax4_all.scatter(removed_points[:, 0], removed_points[:, 1], removed_points[:, 2],
                        c='red', s=10, marker='x', alpha=0.8, label='Removed')
    ax4_all.set_title(f'4. Outlier Removal\n({n_outliers} outliers removed)')
    ax4_all.set_xlabel('X')
    ax4_all.set_ylabel('Y')
    ax4_all.set_zlabel('Z')
    ax4_all.legend()
    
    # Plot 5: Edge weights
    ax5_all = fig_all.add_subplot(2, 4, 5, projection='3d')
    scatter5_all = ax5_all.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                                   c=viz_edge_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax5_all.set_title(f'5. Edge Weights\n({(edge_weights > 0.5).sum()} high-weight points)')
    ax5_all.set_xlabel('X')
    ax5_all.set_ylabel('Y')
    ax5_all.set_zlabel('Z')
    plt.colorbar(scatter5_all, ax=ax5_all, shrink=0.5, label='Edge Weight')
    
    # Plot 6: Cross weights
    ax6_all = fig_all.add_subplot(2, 4, 6, projection='3d')
    scatter6_all = ax6_all.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                                   c=viz_vertex_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax6_all.set_title(f'6. Cross Weights\n({(vertex_weights > 0.5).sum()} high-weight points)')
    ax6_all.set_xlabel('X')
    ax6_all.set_ylabel('Y')
    ax6_all.set_zlabel('Z')
    plt.colorbar(scatter6_all, ax=ax6_all, shrink=0.5, label='Cross Weight')
    
    # Plot 7: Border weights (combined)
    ax7_all = fig_all.add_subplot(2, 4, 7, projection='3d')
    scatter7_all = ax7_all.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                                   c=viz_border_weights, cmap=black_red_yellow, s=3, alpha=0.8, vmin=0, vmax=1)
    ax7_all.set_title(f'7. Border Weights (Combined)\n({(border_weights > 0.5).sum()} high-weight points)')
    ax7_all.set_xlabel('X')
    ax7_all.set_ylabel('Y')
    ax7_all.set_zlabel('Z')
    plt.colorbar(scatter7_all, ax=ax7_all, shrink=0.5, label='Border Weight')
    
    # Plot 8: Final processed point cloud with group colors
    ax8_all = fig_all.add_subplot(2, 4, 8, projection='3d')
    scatter8_all = ax8_all.scatter(viz_filtered_pc[:, 0], viz_filtered_pc[:, 1], viz_filtered_pc[:, 2],
                                   c=viz_filtered_group_ids, cmap='tab20', s=2, alpha=0.7, vmin=0, vmax=n_groups-1)
    ax8_all.set_title(f'8. Final Result\n({len(filtered_pc)} points after preprocessing)')
    ax8_all.set_xlabel('X')
    ax8_all.set_ylabel('Y')
    ax8_all.set_zlabel('Z')
    plt.colorbar(scatter8_all, ax=ax8_all, shrink=0.5, label='Group ID', ticks=range(n_groups))
    
    plt.suptitle(f'Preprocessing Pipeline - {sample_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    
    return fig_all


def select_sample(dataset):
    """Interactive sample selection"""
    print(f"\nAvailable samples: {len(dataset)}")
    print("Sample files:")
    
    # Show available samples
    for i in range(min(10, len(dataset))):
        sample_name = os.path.basename(dataset.pc_files[i]).replace('.xyz', '')
        print(f"  {i}: {sample_name}")
    
    if len(dataset) > 10:
        print(f"  ... and {len(dataset) - 10} more samples")
    
    while True:
        try:
            choice = input(f"\nEnter sample index (0-{len(dataset)-1}): ").strip()
            sample_idx = int(choice)
            
            if 0 <= sample_idx < len(dataset):
                return sample_idx
            else:
                print(f"Please enter a number between 0 and {len(dataset)-1}")
                
        except ValueError:
            print("Please enter a valid number")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualize preprocessing steps')
    parser.add_argument('--sample', type=int, default=None,
                       help='Specific sample index to visualize (if not provided, interactive mode)')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='train',
                       help='Dataset split to use (default: train)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PREPROCESSING VISUALIZATION")
    print("=" * 60)
    
    # Load configuration - handle both relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    config_path = os.path.join(parent_dir, 'datasets', 'dataset_config.yaml')
    
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Fix the root_dir path to be absolute
    if dataset_config.Building3D.root_dir.startswith('./'):
        dataset_config.Building3D.root_dir = os.path.join(parent_dir, dataset_config.Building3D.root_dir[2:])
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    building3D_dataset = build_dataset(dataset_config.Building3D, split_set=args.split)
    
    print(f"Dataset root dir: {dataset_config.Building3D.root_dir}")
    print(f"Loaded {len(building3D_dataset)} samples")
    
    if len(building3D_dataset) == 0:
        print("Error: No samples found in the dataset!")
        return
    
    # Select sample
    if args.sample is not None:
        if args.sample >= len(building3D_dataset):
            print(f"Error: Sample index {args.sample} out of range (0-{len(building3D_dataset)-1})")
            return
        sample_idx = args.sample
    else:
        sample_idx = select_sample(building3D_dataset)
    
    # Load point cloud
    sample_name = os.path.basename(building3D_dataset.pc_files[sample_idx]).replace('.xyz', '')
    print(f"\nLoading sample {sample_idx}: {sample_name}")
    pc = load_point_cloud(building3D_dataset, sample_idx)
    
    # Visualize preprocessing steps
    fig = visualize_preprocessing_steps(pc, dataset_config, sample_name)
    plt.show()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
