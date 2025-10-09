#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-15
# @Author  : Batuhan Arda Bekar
# @File    : surface_grouping.py
# @Purpose : Surface grouping and segmentation for building point clouds

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict
from .outlier_removal import remove_statistical_outliers, remove_radius_outliers, NeighborCache
# Normal estimation removed from pipeline


class SurfaceGroup:
    """
    Represents a group of points belonging to the same surface.
    """
    def __init__(self, points, indices, surface_type='unknown', confidence=0.0):
        self.points = points
        self.indices = indices  # Original indices in the full point cloud
        self.surface_type = surface_type  # 'floor', 'wall', 'roof', 'edge'
        self.confidence = confidence
        self.center = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
        self.size = len(points)
        
    def get_planarity_score(self):
        """Compute planarity score (0=linear, 1=planar)."""
        if len(self.points) < 3:
            return 0.0
            
        centered = self.points - self.center
        cov_matrix = np.cov(centered.T)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        if eigenvalues[0] < 1e-8:
            return 0.0
            
        # Planarity: λ₂/(λ₁+λ₂+λ₃), higher = more planar
        return eigenvalues[1] / (np.sum(eigenvalues) + 1e-8)





def simple_spatial_grouping(points, eps=0.05, min_samples=5, auto_scale_eps=True):
    """
    Perform simple spatial clustering of points without semantic classification.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        eps (float): DBSCAN eps parameter for clustering
        min_samples (int): Minimum samples for DBSCAN
        auto_scale_eps (bool): Auto-scale eps parameter based on point density
        
    Returns:
        list: List of SurfaceGroup objects (one per spatial cluster)
    """
    if len(points) < min_samples:
        return []
    
    # Auto-scale eps if requested
    if auto_scale_eps and len(points) > 5:
        from sklearn.neighbors import NearestNeighbors
        # Use 5th nearest neighbor distance as baseline
        nbrs = NearestNeighbors(n_neighbors=min(6, len(points))).fit(points)
        distances, _ = nbrs.kneighbors(points)
        if distances.shape[1] > 1:
            median_5th_distance = np.median(distances[:, -1])
            eps = max(eps, 1.5 * median_5th_distance)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # Create surface groups
    surface_groups = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
            
        # Get points in this cluster
        cluster_mask = labels == label
        cluster_indices = np.where(cluster_mask)[0]
        cluster_points = points[cluster_mask]
        
        # Create surface group
        group = SurfaceGroup(
            points=cluster_points,
            indices=cluster_indices,
            surface_type='cluster',  # Generic cluster type
            confidence=1.0
        )
        surface_groups.append(group)
    
    return surface_groups



def merge_small_groups_to_nearest(surface_groups, min_group_size):
    """
    Merge small groups (< min_group_size) into their nearest larger groups.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        min_group_size (int): Minimum size threshold for groups
        
    Returns:
        list: Updated list of surface groups with small groups merged
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Separate small and large groups
    small_groups = [g for g in surface_groups if g.size < min_group_size]
    large_groups = [g for g in surface_groups if g.size >= min_group_size]
    
    if not small_groups:
        return surface_groups  # No small groups to merge
    
    if not large_groups:
        # All groups are small, return the largest ones
        surface_groups.sort(key=lambda x: x.size, reverse=True)
        return surface_groups[:max(1, len(surface_groups)//2)]  # Keep at least half
    
    print(f"Merging {len(small_groups)} small groups into {len(large_groups)} larger groups")
    
    # Compute centroids of large groups
    large_centroids = []
    for group in large_groups:
        centroid = np.mean(group.points, axis=0)
        large_centroids.append(centroid)
    
    large_centroids = np.array(large_centroids)
    
    # For each small group, find the nearest large group and merge
    for small_group in small_groups:
        small_centroid = np.mean(small_group.points, axis=0)
        
        # Find nearest large group
        nbrs = NearestNeighbors(n_neighbors=1).fit(large_centroids)
        distances, indices = nbrs.kneighbors([small_centroid])
        nearest_large_idx = indices[0][0]
        
        # Merge small group into nearest large group
        target_group = large_groups[nearest_large_idx]
        
        # Combine points and indices
        combined_points = np.vstack([target_group.points, small_group.points])
        combined_indices = np.concatenate([target_group.indices, small_group.indices])
        
        # Update the target group
        target_group.points = combined_points
        target_group.indices = combined_indices
        target_group.size = len(combined_points)
        
        print(f"  Merged {small_group.size} points into group of {target_group.size - small_group.size} points")
    
    return large_groups


def surface_grouping_pipeline(points, coarse_params=None, refinement_params=None):
    """
    Simple spatial grouping pipeline without semantic classification.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        coarse_params (dict): Parameters for spatial clustering
        refinement_params (dict): Parameters for refinement (optional)
        
    Returns:
        list: Final list of SurfaceGroup objects
    """
    # Default parameters for simple spatial clustering
    if coarse_params is None:
        coarse_params = {
            'eps': 0.05,
            'min_samples': 5,
            'auto_scale_eps': True
        }
    
    print(f"\n=== Simple Spatial Grouping Pipeline ===")
    print(f"Processing {len(points)} points")
    
    # Perform simple spatial clustering
    surface_groups = simple_spatial_grouping(points, **coarse_params)
    
    # Optional: Basic refinement to merge very close groups and small groups
    if refinement_params and len(surface_groups) > 1:
        merge_distance = refinement_params.get('merge_distance', 0.05)
        min_group_size = refinement_params.get('min_group_size', 3)
        
        # Merge small groups into nearest larger groups
        surface_groups = merge_small_groups_to_nearest(surface_groups, min_group_size)
        
        print(f"After merging small groups: {len(surface_groups)} groups")
    
    print(f"\nFinal result: {len(surface_groups)} spatial groups")
    return surface_groups


def get_surface_group_summary(surface_groups):
    """
    Generate a summary of surface groups for analysis.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        
    Returns:
        dict: Summary statistics
    """
    if not surface_groups:
        return {'total_groups': 0, 'total_points': 0}
    
    summary = {
        'total_groups': len(surface_groups),
        'total_points': sum(g.size for g in surface_groups),
        'by_type': {},
        'size_stats': {},
        'confidence_stats': {}
    }
    
    # Group by surface type
    by_type = {}
    for group in surface_groups:
        if group.surface_type not in by_type:
            by_type[group.surface_type] = []
        by_type[group.surface_type].append(group)
    
    # Statistics by type
    for surface_type, groups in by_type.items():
        sizes = [g.size for g in groups]
        confidences = [g.confidence for g in groups]
        
        summary['by_type'][surface_type] = {
            'count': len(groups),
            'total_points': sum(sizes),
            'avg_size': np.mean(sizes),
            'avg_confidence': np.mean(confidences)
        }
    
    # Overall statistics
    all_sizes = [g.size for g in surface_groups]
    all_confidences = [g.confidence for g in surface_groups]
    
    summary['size_stats'] = {
        'min': np.min(all_sizes),
        'max': np.max(all_sizes),
        'mean': np.mean(all_sizes),
        'std': np.std(all_sizes)
    }
    
    summary['confidence_stats'] = {
        'min': np.min(all_confidences),
        'max': np.max(all_confidences),
        'mean': np.mean(all_confidences)
    }
    
    return summary


def get_surface_specific_parameters(surface_type, config):
    """
    Get default parameters for outlier removal and normal computation from config.
    
    Args:
        surface_type (str): Type of surface (not used, kept for compatibility)
        config (dict): Configuration dictionary with default_processing parameters (required)
        
    Returns:
        dict: Default processing parameters from config
    """
    return config['default_processing']


def process_surface_group(surface_group, apply_outlier_removal=True, compute_normals=True, 
                         neighbor_cache=None, custom_outlier_params=None, config=None):
    """
    Process a single surface group with surface-specific outlier removal and normal computation.
    
    Args:
        surface_group (SurfaceGroup): Surface group to process
        apply_outlier_removal (bool): Whether to apply outlier removal
        compute_normals (bool): Whether to compute high-quality normals
        neighbor_cache (NeighborCache): Optional neighbor cache for performance
        custom_outlier_params (dict): Custom outlier removal parameters (optional)
        config (dict): Configuration dictionary with processing parameters (required)
        
    Returns:
        tuple: (cleaned_points, cleaned_indices, normals, quality_scores)
    """
    if surface_group.size < 5:
        # Too small for meaningful processing
        return surface_group.points, surface_group.indices, None, None
    
    if config is None:
        raise ValueError("config parameter is required")
    
    # Get surface-specific parameters
    params = get_surface_specific_parameters(surface_group.surface_type, config)
    
    # Override with custom parameters if provided
    if custom_outlier_params is not None and apply_outlier_removal:
        # Merge custom parameters with defaults
        outlier_params = params['outlier_removal'].copy()
        outlier_params.update(custom_outlier_params)
        params['outlier_removal'] = outlier_params
    
    current_points = surface_group.points.copy()
    current_indices = surface_group.indices.copy()
    
    # Phase 1: Surface-specific outlier removal
    if apply_outlier_removal and len(current_points) > 10:
        outlier_params = params['outlier_removal']
        
        # Statistical outlier removal
        clean_points, stat_mask = remove_statistical_outliers(
            current_points, 
            k_neighbors=outlier_params['stat_k'],
            std_ratio=outlier_params['stat_std_ratio']
        )
        
        if len(clean_points) > 5:
            # Update indices and points
            current_indices = current_indices[stat_mask]
            current_points = clean_points
            
            # Radius outlier removal
            clean_points, radius_mask = remove_radius_outliers(
                current_points,
                radius=outlier_params['radius_threshold'],
                min_neighbors=outlier_params['radius_min_neighbors'],
                auto_scale_radius_param=outlier_params['auto_scale_radius_param']
            )
            
            if len(clean_points) > 3:
                current_indices = current_indices[radius_mask]
                current_points = clean_points
    
    # Phase 2: Normals removed from pipeline
    normals = None
    quality_scores = None
    
    return current_points, current_indices, normals, quality_scores


def process_all_surface_groups(surface_groups, apply_outlier_removal=True, compute_normals=True,
                              use_neighbor_cache=True, outlier_removal_params=None, config=None):
    """
    Process all surface groups with surface-specific parameters.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        apply_outlier_removal (bool): Whether to apply outlier removal
        compute_normals (bool): Whether to compute high-quality normals
        use_neighbor_cache (bool): Whether to use neighbor caching for performance
        outlier_removal_params (dict): Custom outlier removal parameters (optional)
        config (dict): Configuration dictionary with processing parameters (required)
        
    Returns:
        dict: Processed results with structure:
            {
                'cleaned_points': np.ndarray,     # All cleaned points
                'cleaned_indices': np.ndarray,    # Original indices of cleaned points  
                'normals': np.ndarray,            # Computed normals
                'quality_scores': np.ndarray,     # Normal quality scores
                'surface_labels': np.ndarray,     # Surface type labels for each point
                'group_info': list                # Per-group processing info
            }
    """
    if config is None:
        raise ValueError("config parameter is required")
    
    # Initialize neighbor cache if requested
    neighbor_cache = NeighborCache() if use_neighbor_cache else None
    
    all_cleaned_points = []
    all_cleaned_indices = []
    all_quality_scores = []
    all_group_ids = []
    group_info = []
    
    for i, group in enumerate(surface_groups):
        # Process this group
        cleaned_points, cleaned_indices, normals, quality_scores = process_surface_group(
            group, apply_outlier_removal, compute_normals, neighbor_cache, outlier_removal_params, config
        )
        
        n_original = group.size
        n_cleaned = len(cleaned_points)
        retention_rate = n_cleaned / n_original if n_original > 0 else 0.0
        
        group_info.append({
            'group_id': i,
            'surface_type': 'cluster',
            'original_size': n_original,
            'cleaned_size': n_cleaned,
            'retention_rate': retention_rate,
            'has_normals': False
        })
        
        if n_cleaned > 0:
            # Store results
            all_cleaned_points.append(cleaned_points)
            all_cleaned_indices.append(cleaned_indices)
            
            # Group ids (unique per group)
            group_ids = np.full(n_cleaned, i, dtype=np.int32)
            all_group_ids.append(group_ids)
            
            # Quality scores (default since normals removed)
            default_quality = np.full(n_cleaned, 0.5)
            all_quality_scores.append(default_quality)
            
    # Combine all results
    if all_cleaned_points:
        combined_points = np.vstack(all_cleaned_points)
        combined_indices = np.concatenate(all_cleaned_indices)
        combined_quality = np.concatenate(all_quality_scores) if all_quality_scores else None
        combined_group_ids = np.concatenate(all_group_ids) if all_group_ids else None
    else:
        combined_points = np.zeros((0, 3))
        combined_indices = np.array([], dtype=np.int32)
        combined_quality = None
        combined_group_ids = None

    results = {
        'cleaned_points': combined_points,
        'cleaned_indices': combined_indices,
        'quality_scores': combined_quality,
        'group_ids': combined_group_ids,
        'group_info': group_info
    }
    
    return results


def surface_aware_normalization_pipeline(points, outlier_removal_params=None, 
                                        grouping_params=None, processing_params=None):
    """
    Complete surface-aware processing pipeline combining all components.
    Modified to group first, then apply outlier removal per group.
    
    Args:
        points (np.ndarray): Input point cloud coordinates with shape (N, 3)
        outlier_removal_params (dict): Parameters for outlier removal (applied per group)
        grouping_params (dict): Parameters for surface grouping
        processing_params (dict): Parameters for per-surface processing
        
    Returns:
        dict: Complete processing results including cleaned points, normals, and surface labels
    """
    print(f"\n" + "="*60)
    print(f"SURFACE-AWARE NORMALIZATION PIPELINE")
    print(f"="*60)
    print(f"Input: {len(points)} points")
    
    # Step 1: Surface grouping on raw points (no initial outlier removal)
    if grouping_params is None:
        grouping_params = {}
    
    print(f"\nStep 1: Surface grouping")
    surface_groups = surface_grouping_pipeline(points, 
                                             coarse_params=grouping_params.get('coarse_params'),
                                             refinement_params=grouping_params.get('refinement_params'))
    
    if not surface_groups:
        print("Warning: No surface groups detected!")
        return None
    
    # Step 2: Per-surface processing (includes outlier removal per group)
    if processing_params is None:
        processing_params = {
            'apply_outlier_removal': True,
            'compute_normals': True,
            'use_neighbor_cache': True
        }
    
    # Pass outlier removal parameters to per-group processing
    if outlier_removal_params is not None:
        processing_params['outlier_removal_params'] = outlier_removal_params
    
    # Pass grouping config to processing
    processing_params['config'] = grouping_params
    
    print(f"\nStep 2: Per-surface processing")
    processing_results = process_all_surface_groups(surface_groups, **processing_params)
    
    # Step 3: Combine results
    final_results = {
        'original_points': points,
        'cleaned_points': processing_results['cleaned_points'],
        'quality_scores': processing_results['quality_scores'],
        'group_ids': processing_results['group_ids'],
        'surface_groups': surface_groups,
        'group_info': processing_results['group_info'],
        'outlier_mask': None,  # No global outlier removal
        'processing_summary': get_processing_summary(points, processing_results, surface_groups)
    }
    
    return final_results

def get_processing_summary(original_points, processing_results, surface_groups):
    """
    Generate a comprehensive summary of the processing pipeline.
    """
    n_original = len(original_points)
    n_final = len(processing_results['cleaned_points'])
    retention_rate = n_final / n_original if n_original > 0 else 0.0
    
    # Group statistics
    group_stats = {}
    if processing_results['group_ids'] is not None:
        unique_groups, counts = np.unique(processing_results['group_ids'], return_counts=True)
        
        for group_id, count in zip(unique_groups, counts):
            group_stats[f'group_{group_id}'] = {
                'points': count,
                'percentage': 100 * count / n_final if n_final > 0 else 0.0
            }
    
    summary = {
        'total_original_points': n_original,
        'total_final_points': n_final,
        'overall_retention_rate': retention_rate,
        'total_surface_groups': len(surface_groups),
        'group_statistics': group_stats,
        'has_quality_scores': processing_results['quality_scores'] is not None
    }
    
    return summary