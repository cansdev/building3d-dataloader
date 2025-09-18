#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-15
# @Author  : GitHub Copilot
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


def height_stratification(points, floor_height_range=(0.0, 0.3), wall_height_range=(0.3, 0.85), 
                         roof_height_min=0.85, normalize_heights=True):
    """
    Stratify points by height into floor, wall, and roof categories.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        floor_height_range (tuple): Relative height range for floors (0-1)
        wall_height_range (tuple): Relative height range for walls (0-1)
        roof_height_min (float): Minimum relative height for roofs (0-1)
        normalize_heights (bool): Whether to normalize heights to [0,1] range
        
    Returns:
        dict: {'floor': indices, 'wall': indices, 'roof': indices}
    """
    z_coords = points[:, 2]
    
    if normalize_heights:
        # Normalize heights to [0, 1] range
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        z_range = z_max - z_min
        if z_range > 1e-6:
            z_normalized = (z_coords - z_min) / z_range
        else:
            z_normalized = np.zeros_like(z_coords)
    else:
        z_normalized = z_coords
    
    # Classify by height
    floor_mask = (z_normalized >= floor_height_range[0]) & (z_normalized <= floor_height_range[1])
    wall_mask = (z_normalized > wall_height_range[0]) & (z_normalized <= wall_height_range[1])
    roof_mask = z_normalized > roof_height_min
    
    # Handle overlaps (prioritize: floor > wall > roof)
    wall_mask = wall_mask & ~floor_mask
    roof_mask = roof_mask & ~floor_mask & ~wall_mask
    
    strata = {
        'floor': np.where(floor_mask)[0],
        'wall': np.where(wall_mask)[0], 
        'roof': np.where(roof_mask)[0]
    }
    
    print(f"Height stratification results:")
    print(f"  Floor: {len(strata['floor'])} points ({100*len(strata['floor'])/len(points):.1f}%)")
    print(f"  Wall: {len(strata['wall'])} points ({100*len(strata['wall'])/len(points):.1f}%)")
    print(f"  Roof: {len(strata['roof'])} points ({100*len(strata['roof'])/len(points):.1f}%)")
    
    return strata


def spatial_clustering_within_strata(points, strata_indices, eps=0.05, min_samples=5, 
                                   auto_scale_eps=True):
    """
    Perform spatial clustering within each height stratum.
    
    Args:
        points (np.ndarray): Full point cloud coordinates
        strata_indices (np.ndarray): Indices of points in this stratum
        eps (float): DBSCAN eps parameter (auto-scaled if enabled)
        min_samples (int): DBSCAN min_samples parameter
        auto_scale_eps (bool): Whether to auto-scale eps based on point density
        
    Returns:
        list: List of SurfaceGroup objects
    """
    if len(strata_indices) < min_samples:
        return []
    
    stratum_points = points[strata_indices]
    
    # Auto-scale eps if requested
    if auto_scale_eps and len(stratum_points) > 10:
        # Use median distance to 5th nearest neighbor
        k = min(5, len(stratum_points) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(stratum_points)
        distances, _ = nbrs.kneighbors(stratum_points)
        median_5th_distance = np.median(distances[:, -1])
        eps = max(eps, 1.5 * median_5th_distance)
    
    # Perform DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(stratum_points)
    labels = clustering.labels_
    
    # Create surface groups
    surface_groups = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label == -1:  # Noise points
            continue
            
        # Get points in this cluster
        cluster_mask = labels == label
        cluster_indices = strata_indices[cluster_mask]
        cluster_points = stratum_points[cluster_mask]
        
        # Create surface group
        group = SurfaceGroup(
            points=cluster_points,
            indices=cluster_indices,
            surface_type='unknown',
            confidence=0.5
        )
        surface_groups.append(group)
    
    print(f"  Spatial clustering: {len(surface_groups)} groups found (eps={eps:.4f})")
    
    return surface_groups


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
    print(f"\n=== Simple Spatial Grouping ===")
    print(f"Input: {len(points)} points")
    
    if len(points) < min_samples:
        print(f"Too few points ({len(points)}) for clustering")
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
            print(f"Auto-scaled eps to {eps:.4f}")
    
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
    
    print(f"Found {len(surface_groups)} spatial clusters (eps={eps:.4f})")
    
    return surface_groups


def coarse_geometric_grouping(points, floor_height_range=(0.0, 0.3), wall_height_range=(0.3, 0.85),
                             roof_height_min=0.85, eps_floor=0.05, eps_wall=0.03, eps_roof=0.08,
                             min_samples=5, auto_scale_eps=True):
    """
    Perform coarse geometric grouping based on height stratification and spatial clustering.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        floor_height_range (tuple): Height range for floor classification
        wall_height_range (tuple): Height range for wall classification  
        roof_height_min (float): Minimum height for roof classification
        eps_floor (float): DBSCAN eps for floor clustering
        eps_wall (float): DBSCAN eps for wall clustering
        eps_roof (float): DBSCAN eps for roof clustering
        min_samples (int): Minimum samples for DBSCAN
        auto_scale_eps (bool): Auto-scale eps parameters
        
    Returns:
        list: List of SurfaceGroup objects
    """
    print(f"\n=== Coarse Geometric Grouping ===")
    
    # Step 1: Height stratification
    strata = height_stratification(points, floor_height_range, wall_height_range, roof_height_min)
    
    # Step 2: Spatial clustering within each stratum
    all_groups = []
    
    # Floor clustering
    if len(strata['floor']) > 0:
        print(f"Clustering floor points...")
        floor_groups = spatial_clustering_within_strata(
            points, strata['floor'], eps_floor, min_samples, auto_scale_eps
        )
        for group in floor_groups:
            group.surface_type = 'floor'
            group.confidence = 0.8  # High confidence for floors
        all_groups.extend(floor_groups)
    
    # Wall clustering 
    if len(strata['wall']) > 0:
        print(f"Clustering wall points...")
        wall_groups = spatial_clustering_within_strata(
            points, strata['wall'], eps_wall, min_samples, auto_scale_eps
        )
        for group in wall_groups:
            group.surface_type = 'wall'
            group.confidence = 0.7  # Medium-high confidence for walls
        all_groups.extend(wall_groups)
    
    # Roof clustering
    if len(strata['roof']) > 0:
        print(f"Clustering roof points...")
        roof_groups = spatial_clustering_within_strata(
            points, strata['roof'], eps_roof, min_samples, auto_scale_eps
        )
        for group in roof_groups:
            group.surface_type = 'roof' 
            group.confidence = 0.6  # Lower confidence for roofs (more complex)
        all_groups.extend(roof_groups)
    
    # Compute dominant normals for each group
    for group in all_groups:
        group.compute_dominant_normal()
    
    print(f"Coarse grouping complete: {len(all_groups)} surface groups")
    return all_groups


def normal_similarity(normal1, normal2, angle_threshold=30.0):
    """
    Compute similarity between two normal vectors.
    
    Args:
        normal1, normal2 (np.ndarray): Unit normal vectors
        angle_threshold (float): Maximum angle difference in degrees
        
    Returns:
        float: Similarity score (0-1, higher = more similar)
    """
    # Compute angle between normals
    dot_product = np.clip(np.dot(normal1, normal2), -1.0, 1.0)
    angle_rad = np.arccos(abs(dot_product))  # Use abs for bi-directional normals
    angle_deg = np.degrees(angle_rad)
    
    if angle_deg <= angle_threshold:
        return 1.0 - (angle_deg / angle_threshold)
    else:
        return 0.0


def should_merge_groups(group1, group2, max_distance=0.1, min_normal_similarity=0.7,
                       same_type_bonus=0.2):
    """
    Determine if two surface groups should be merged.
    
    Args:
        group1, group2 (SurfaceGroup): Groups to consider merging
        max_distance (float): Maximum distance between group centers
        min_normal_similarity (float): Minimum normal similarity for merging
        same_type_bonus (float): Bonus similarity for same surface type
        
    Returns:
        bool: Whether groups should be merged
    """
    # Check distance between centers
    center_distance = np.linalg.norm(group1.center - group2.center)
    if center_distance > max_distance:
        return False
    
    # Check normal similarity
    if group1.normal is None:
        group1.compute_dominant_normal()
    if group2.normal is None:
        group2.compute_dominant_normal()
        
    normal_sim = normal_similarity(group1.normal, group2.normal)
    
    # Bonus for same surface type
    if group1.surface_type == group2.surface_type:
        normal_sim += same_type_bonus
    
    return normal_sim >= min_normal_similarity


def should_split_group(group, normal_angle_threshold=45.0, min_split_size=10):
    """
    Determine if a surface group should be split based on normal variation.
    
    Args:
        group (SurfaceGroup): Group to consider splitting
        normal_angle_threshold (float): Max angle variation within group
        min_split_size (int): Minimum size for split sub-groups
        
    Returns:
        bool: Whether group should be split
    """
    if group.size < 2 * min_split_size:
        return False
    
    # Compute normals for subsets of points
    n_points = len(group.points)
    k = min(10, n_points // 3)  # Sample size for local normal computation
    
    if n_points < k:
        return False
    
    # Sample multiple subsets and compute their normals
    normals = []
    for _ in range(min(5, n_points // k)):
        indices = np.random.choice(n_points, k, replace=False)
        subset = group.points[indices]
        
        # Compute normal for subset
        centered = subset - np.mean(subset, axis=0)
        if len(centered) >= 3:
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0]  # Smallest eigenvalue
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            normals.append(normal)
    
    if len(normals) < 2:
        return False
    
    # Check variation in normals
    max_angle = 0.0
    for i in range(len(normals)):
        for j in range(i + 1, len(normals)):
            dot_product = np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(dot_product)))
            max_angle = max(max_angle, angle)
    
    return max_angle > normal_angle_threshold


def refine_groups_with_normals(surface_groups, merge_distance=0.1, merge_normal_threshold=0.7,
                              split_angle_threshold=45.0, min_group_size=5):
    """
    Refine surface groups using normal information to merge similar groups and split mixed groups.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        merge_distance (float): Maximum distance for merging groups
        merge_normal_threshold (float): Minimum normal similarity for merging
        split_angle_threshold (float): Maximum angle variation within groups
        min_group_size (int): Minimum group size for operations
        
    Returns:
        list: Refined list of SurfaceGroup objects
    """
    print(f"\n=== Normal-Guided Refinement ===")
    print(f"Starting with {len(surface_groups)} groups")
    
    refined_groups = surface_groups.copy()
    
    # Phase 1: Merge similar adjacent groups
    merged_count = 0
    i = 0
    while i < len(refined_groups):
        group1 = refined_groups[i]
        if group1.size < min_group_size:
            i += 1
            continue
            
        j = i + 1
        while j < len(refined_groups):
            group2 = refined_groups[j]
            
            if should_merge_groups(group1, group2, merge_distance, merge_normal_threshold):
                # Merge groups
                merged_points = np.vstack([group1.points, group2.points])
                merged_indices = np.concatenate([group1.indices, group2.indices])
                
                merged_group = SurfaceGroup(
                    points=merged_points,
                    indices=merged_indices,
                    surface_type=group1.surface_type,  # Keep first group's type
                    confidence=max(group1.confidence, group2.confidence)
                )
                merged_group.compute_dominant_normal()
                
                # Replace first group and remove second
                refined_groups[i] = merged_group
                refined_groups.pop(j)
                merged_count += 1
            else:
                j += 1
        i += 1
    
    print(f"Merged {merged_count} group pairs")
    
    # Phase 2: Split groups with high normal variation
    split_count = 0
    groups_to_split = []
    
    for i, group in enumerate(refined_groups):
        if should_split_group(group, split_angle_threshold, min_group_size):
            groups_to_split.append(i)
    
    # Split identified groups (process in reverse order to maintain indices)
    for i in reversed(groups_to_split):
        group = refined_groups[i]
        
        # Simple split: cluster by normal direction
        if len(group.points) >= 2 * min_group_size:
            # Compute local normals for each point
            k = min(10, len(group.points) // 3)
            nbrs = NearestNeighbors(n_neighbors=k).fit(group.points)
            _, indices = nbrs.kneighbors(group.points)
            
            point_normals = []
            for j, point_indices in enumerate(indices):
                neighborhood = group.points[point_indices]
                centered = neighborhood - np.mean(neighborhood, axis=0)
                cov_matrix = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                normal = eigenvectors[:, 0]
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                point_normals.append(normal)
            
            point_normals = np.array(point_normals)
            
            # Cluster normals using DBSCAN
            normal_clustering = DBSCAN(eps=0.3, min_samples=min_group_size).fit(point_normals)
            normal_labels = normal_clustering.labels_
            
            unique_labels = np.unique(normal_labels)
            if len(unique_labels) > 1 and -1 not in unique_labels:  # Successful split
                # Create new groups
                new_groups = []
                for label in unique_labels:
                    if label == -1:
                        continue
                    mask = normal_labels == label
                    if np.sum(mask) >= min_group_size:
                        new_group = SurfaceGroup(
                            points=group.points[mask],
                            indices=group.indices[mask],
                            surface_type=group.surface_type,
                            confidence=group.confidence * 0.8  # Slightly lower confidence
                        )
                        new_group.compute_dominant_normal()
                        new_groups.append(new_group)
                
                if len(new_groups) > 1:
                    # Replace original group with split groups
                    refined_groups[i:i+1] = new_groups
                    split_count += 1
    
    print(f"Split {split_count} groups")
    
    # Remove very small groups
    initial_count = len(refined_groups)
    refined_groups = [g for g in refined_groups if g.size >= min_group_size]
    removed_count = initial_count - len(refined_groups)
    
    if removed_count > 0:
        print(f"Removed {removed_count} small groups (< {min_group_size} points)")
    
    print(f"Refinement complete: {len(refined_groups)} final groups")
    return refined_groups


def classify_surface_types(surface_groups, wall_normal_z_threshold=0.3, 
                          floor_normal_z_threshold=0.8):
    """
    Classify or reclassify surface types based on computed normals.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        wall_normal_z_threshold (float): Max Z-component for wall normals
        floor_normal_z_threshold (float): Min Z-component for floor/roof normals
        
    Returns:
        list: Surface groups with updated classifications
    """
    print(f"\n=== Surface Type Classification ===")
    
    type_counts = {'floor': 0, 'wall': 0, 'roof': 0, 'edge': 0, 'unknown': 0}
    
    for group in surface_groups:
        if group.normal is None:
            group.compute_dominant_normal()
        
        z_component = abs(group.normal[2])
        avg_height = np.mean(group.points[:, 2])
        
        # Classify based on normal orientation and height
        if z_component >= floor_normal_z_threshold:
            # Horizontal surface - floor or roof based on height
            if avg_height < 0.5:  # Relative to normalized range
                group.surface_type = 'floor'
                group.confidence = 0.9
            else:
                group.surface_type = 'roof'
                group.confidence = 0.8
        elif z_component <= wall_normal_z_threshold:
            # Vertical surface - wall
            group.surface_type = 'wall'  
            group.confidence = 0.8
        else:
            # Oblique surface - could be edge or complex geometry
            if group.size < 20:
                group.surface_type = 'edge'
                group.confidence = 0.5
            else:
                group.surface_type = 'roof'  # Likely sloped roof
                group.confidence = 0.6
        
        type_counts[group.surface_type] += 1
    
    print(f"Surface classification results:")
    for surface_type, count in type_counts.items():
        if count > 0:
            print(f"  {surface_type.capitalize()}: {count} groups")
    
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


def get_surface_specific_parameters(surface_type):
    """
    Get surface-specific parameters for outlier removal and normal computation.
    
    Args:
        surface_type (str): Type of surface ('floor', 'wall', 'roof', 'edge')
        
    Returns:
        dict: Surface-specific parameters
    """
    if surface_type == 'floor':
        return {
            'outlier_removal': {
                'stat_k': 15,
                'stat_std_ratio': 1.8,
                'radius_threshold': 0.03,
                'radius_min_neighbors': 4,
                'auto_scale_radius_param': True
            },
            'normal_computation': {
                'base_k_neighbors': 15,
                'curvature_threshold': 0.05,  # Floors should be very planar
                'smooth_iterations': 2,
                'smooth_radius': 0.04,
                'adaptive_k': True
            }
        }
    elif surface_type == 'wall':
        return {
            'outlier_removal': {
                'stat_k': 12,
                'stat_std_ratio': 2.0,
                'radius_threshold': 0.025,
                'radius_min_neighbors': 3,
                'auto_scale_radius_param': True
            },
            'normal_computation': {
                'base_k_neighbors': 12,
                'curvature_threshold': 0.08,  # Walls should be planar
                'smooth_iterations': 1,
                'smooth_radius': 0.03,
                'adaptive_k': True
            }
        }
    elif surface_type == 'roof':
        return {
            'outlier_removal': {
                'stat_k': 10,
                'stat_std_ratio': 2.5,  # More permissive for complex roof shapes
                'radius_threshold': 0.04,
                'radius_min_neighbors': 2,
                'auto_scale_radius_param': True
            },
            'normal_computation': {
                'base_k_neighbors': 10,
                'curvature_threshold': 0.15,  # Roofs can be more complex
                'smooth_iterations': 1,
                'smooth_radius': 0.05,
                'adaptive_k': True
            }
        }
    elif surface_type == 'edge':
        return {
            'outlier_removal': {
                'stat_k': 8,
                'stat_std_ratio': 3.0,  # Very permissive for edge regions
                'radius_threshold': 0.02,
                'radius_min_neighbors': 2,
                'auto_scale_radius_param': True
            },
            'normal_computation': {
                'base_k_neighbors': 8,
                'curvature_threshold': 0.3,  # Edges can be highly curved
                'smooth_iterations': 0,  # No smoothing for edges
                'smooth_radius': 0.02,
                'adaptive_k': True
            }
        }
    else:  # unknown or default
        return {
            'outlier_removal': {
                'stat_k': 12,
                'stat_std_ratio': 3.5,  # More conservative (was 2.0)
                'radius_threshold': 0.03,
                'radius_min_neighbors': 2,  # More lenient (was 3)
                'auto_scale_radius_param': True
            },
            'normal_computation': {
                'base_k_neighbors': 12,
                'curvature_threshold': 0.1,
                'smooth_iterations': 1,
                'smooth_radius': 0.04,
                'adaptive_k': True
            }
        }


def process_surface_group(surface_group, apply_outlier_removal=True, compute_normals=True, 
                         neighbor_cache=None, custom_outlier_params=None):
    """
    Process a single surface group with surface-specific outlier removal and normal computation.
    
    Args:
        surface_group (SurfaceGroup): Surface group to process
        apply_outlier_removal (bool): Whether to apply outlier removal
        compute_normals (bool): Whether to compute high-quality normals
        neighbor_cache (NeighborCache): Optional neighbor cache for performance
        custom_outlier_params (dict): Custom outlier removal parameters (optional)
        
    Returns:
        tuple: (cleaned_points, cleaned_indices, normals, quality_scores)
    """
    if surface_group.size < 5:
        # Too small for meaningful processing
        return surface_group.points, surface_group.indices, None, None
    
    # Get surface-specific parameters
    params = get_surface_specific_parameters(surface_group.surface_type)
    
    # Override with custom parameters if provided
    if custom_outlier_params is not None and apply_outlier_removal:
        # Merge custom parameters with defaults
        outlier_params = params['outlier_removal'].copy()
        outlier_params.update(custom_outlier_params)
        params['outlier_removal'] = outlier_params
    
    current_points = surface_group.points.copy()
    current_indices = surface_group.indices.copy()
    
    print(f"Processing {surface_group.surface_type} surface ({surface_group.size} points)")
    
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
    
    # Phase 2: High-quality normal computation
    normals = None
    quality_scores = None
    
    if compute_normals and len(current_points) >= 3:
        # Normal computation disabled - normals removed from pipeline
        print("  Normal computation disabled (normals removed from pipeline)")
        normals = None
        quality_scores = None
    
    return current_points, current_indices, normals, quality_scores


def process_all_surface_groups(surface_groups, apply_outlier_removal=True, compute_normals=True,
                              use_neighbor_cache=True, outlier_removal_params=None):
    """
    Process all surface groups with surface-specific parameters.
    
    Args:
        surface_groups (list): List of SurfaceGroup objects
        apply_outlier_removal (bool): Whether to apply outlier removal
        compute_normals (bool): Whether to compute high-quality normals
        use_neighbor_cache (bool): Whether to use neighbor caching for performance
        outlier_removal_params (dict): Custom outlier removal parameters (optional)
        
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
    print(f"\n=== Per-Surface Processing ===")
    print(f"Processing {len(surface_groups)} surface groups")
    
    # Initialize neighbor cache if requested
    neighbor_cache = NeighborCache() if use_neighbor_cache else None
    
    all_cleaned_points = []
    all_cleaned_indices = []
    all_quality_scores = []
    all_group_ids = []
    group_info = []
    
    for i, group in enumerate(surface_groups):
        print(f"\nGroup {i+1}/{len(surface_groups)}: cluster {i}")
        
        # Process this group
        cleaned_points, cleaned_indices, normals, quality_scores = process_surface_group(
            group, apply_outlier_removal, compute_normals, neighbor_cache, outlier_removal_params
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
            'has_normals': normals is not None
        })
        
        print(f"  Retained {n_cleaned}/{n_original} points ({100*retention_rate:.1f}%)")
        
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
    
    print(f"\n=== Processing Summary ===")
    print(f"Total cleaned points: {len(combined_points)}")
    
    # Summary by groups
    total_groups = len(group_info)
    total_points = sum(info['cleaned_size'] for info in group_info)
    
    print(f"  Total: {total_groups} spatial groups, {total_points} points")
    
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