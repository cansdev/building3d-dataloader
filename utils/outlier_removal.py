#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-14
# @Author  : Batuhan Arda Bekar
# @File    : outlier_removal.py
# @Purpose : Point cloud outlier detection and removal utilities

import numpy as np
from sklearn.neighbors import NearestNeighbors


class NeighborCache:
    """
    Cache for NearestNeighbors trees to avoid rebuilding in chained operations.
    """
    def __init__(self):
        self.tree = None
        self.points = None
        self.points_hash = None
    
    def get_neighbors(self, points, algorithm='kd_tree'):
        """
        Get or create NearestNeighbors tree for the given points.
        
        Args:
            points (np.ndarray): Point cloud coordinates
            algorithm (str): Algorithm for neighbor search
            
        Returns:
            NearestNeighbors: Fitted neighbor tree
        """
        # Create a simple hash for the points array
        current_hash = hash(points.data.tobytes())
        
        if (self.tree is None or 
            self.points_hash != current_hash or 
            self.points is None or 
            not np.array_equal(self.points, points)):
            
            # Rebuild tree
            self.tree = NearestNeighbors(algorithm=algorithm).fit(points)
            self.points = points.copy()
            self.points_hash = current_hash
            
        return self.tree


def auto_scale_radius(points, base_radius=0.05, k_sample=16, scale_factor=1.5):
    """
    Automatically scale radius based on local point cloud density.
    Prevents issues when radius is inappropriate for the cloud's scale.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        base_radius (float): Base radius (used if auto-scaling fails)
        k_sample (int): Number of neighbors to sample for distance estimation
        scale_factor (float): Multiplier for median k-th neighbor distance
        
    Returns:
        float: Automatically scaled radius
    """
    n_points = points.shape[0]
    
    if n_points < k_sample + 1:
        # Not enough points for auto-scaling
        return base_radius
    
    try:
        # Sample k-th nearest neighbor distances
        k_neighbors = min(k_sample, n_points - 1)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree').fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        # Use median k-th neighbor distance as scale reference
        kth_distances = distances[:, -1]  # k-th neighbor distance for each point
        median_kth_distance = np.median(kth_distances)
        
        # Scale the radius
        auto_radius = scale_factor * median_kth_distance
        
        # Sanity check: ensure radius is reasonable
        if auto_radius > 0 and auto_radius < 10 * median_kth_distance:
            return auto_radius
        else:
            return base_radius
            
    except Exception:
        # Fallback to base radius if auto-scaling fails
        return base_radius


def remove_statistical_outliers(points, k_neighbors=20, std_ratio=2.0):
    """
    Remove statistical outliers using mean distance to k-nearest neighbors.
    Points whose mean distance is beyond std_ratio * standard_deviation are removed.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        k_neighbors (int): Number of neighbors to compute mean distance
        std_ratio (float): Standard deviation multiplier for outlier threshold
        
    Returns:
        tuple: (inlier_points, inlier_mask) where inlier_mask is boolean array
    """
    n_points = points.shape[0]
    
    # Handle edge cases with very small point clouds
    if n_points < 2:
        return points.copy(), np.ones(n_points, dtype=bool)
    
    # Clamp k_neighbors to valid range
    k_neighbors = min(k_neighbors, max(1, n_points - 1))
    
    # Find k-nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='kd_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Compute mean distance to neighbors (excluding self at index 0)
    mean_distances = np.mean(distances[:, 1:], axis=1)
    
    # Compute global statistics
    global_mean = np.mean(mean_distances)
    global_std = np.std(mean_distances)
    
    # Identify inliers
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances <= threshold
    
    inlier_points = points[inlier_mask]
    
    return inlier_points, inlier_mask


def remove_radius_outliers(points, radius=0.05, min_neighbors=2, auto_scale_radius_param=True):
    """
    Remove radius outliers - points that have fewer than min_neighbors within radius.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        radius (float): Search radius for neighbor counting (auto-scaled if enabled)
        min_neighbors (int): Minimum number of neighbors required
        auto_scale_radius_param (bool): Whether to auto-scale radius based on point density
        
    Returns:
        tuple: (inlier_points, inlier_mask) where inlier_mask is boolean array
    """
    n_points = points.shape[0]
    
    # Handle edge cases with very small point clouds
    if n_points < 2:
        return points.copy(), np.ones(n_points, dtype=bool)
    
    # Auto-scale radius if requested
    if auto_scale_radius_param:
        original_radius = radius
        radius = auto_scale_radius(points, base_radius=radius)
        if abs(radius - original_radius) > 0.001:
            print(f"  Auto-scaled radius: {original_radius:.4f} → {radius:.4f}")
    
    # Find neighbors within radius for each point
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(points)
    
    # Vectorized radius neighbor search - much faster than Python loop
    neighbor_indices = nbrs.radius_neighbors(points, radius=radius, return_distance=False)
    
    # Count neighbors for each point (excluding self)
    neighbor_counts = np.fromiter((len(indices) - 1 for indices in neighbor_indices), 
                                  dtype=np.int32, count=n_points)
    
    # Create inlier mask
    inlier_mask = neighbor_counts >= min_neighbors
    
    inlier_points = points[inlier_mask]
    
    return inlier_points, inlier_mask


def remove_elevation_outliers(points, z_std_ratio=2.5, percentile_range=(5, 95), use_dominant_plane=False):
    """
    Remove elevation outliers - points significantly above or below the main surface.
    Uses robust statistics to handle building surfaces with multiple levels.
    Optionally detects dominant plane orientation instead of assuming Z=up.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        z_std_ratio (float): Standard deviation multiplier for outlier threshold
        percentile_range (tuple): Percentile range to compute robust statistics
        use_dominant_plane (bool): Whether to detect dominant plane via PCA
        
    Returns:
        tuple: (inlier_points, inlier_mask) where inlier_mask is boolean array
    """
    n_points = points.shape[0]
    
    # Handle edge cases with very small point clouds
    if n_points < 2:
        return points.copy(), np.ones(n_points, dtype=bool)
    
    if use_dominant_plane and n_points >= 10:
        # Use PCA to find dominant plane orientation
        inlier_points, inlier_mask = _remove_plane_distance_outliers(
            points, z_std_ratio, percentile_range
        )
        print(f"Plane-based outlier removal: {np.sum(~inlier_mask)}/{n_points} points removed ({100*np.sum(~inlier_mask)/n_points:.1f}%)")
        return inlier_points, inlier_mask
    else:
        # Traditional Z-coordinate based removal
        return _remove_z_elevation_outliers(points, z_std_ratio, percentile_range)


def _remove_plane_distance_outliers(points, distance_std_ratio=2.5, percentile_range=(5, 95)):
    """
    Remove outliers based on distance to dominant plane detected via PCA.
    Works for any orientation and handles tilted or sloped surfaces.
    """
    n_points = points.shape[0]
    
    # Find dominant plane using PCA on a subset of points (for robustness)
    subset_size = min(1000, n_points)  # Use subset for large clouds
    if n_points > subset_size:
        # Sample points for plane fitting (use middle percentiles for robustness)
        subset_indices = np.random.choice(n_points, subset_size, replace=False)
        subset_points = points[subset_indices]
    else:
        subset_points = points
    
    # Center the points
    centroid = np.mean(subset_points, axis=0)
    centered_points = subset_points - centroid
    
    # Compute covariance matrix and find eigenvectors
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort by eigenvalues (ascending), normal is eigenvector of smallest eigenvalue
    sorted_indices = np.argsort(eigenvalues)
    plane_normal = eigenvectors[:, sorted_indices[0]]  # Smallest eigenvalue
    
    # Ensure consistent normal orientation (point away from centroid majority)
    test_point = centroid + 0.1 * plane_normal
    distances_to_test = np.dot(points - test_point, plane_normal)
    if np.mean(distances_to_test > 0) > 0.5:
        plane_normal = -plane_normal
    
    # Compute distance from each point to the dominant plane
    plane_distances = np.abs(np.dot(points - centroid, plane_normal))
    
    # Use robust statistics for distance threshold
    low_pct, high_pct = percentile_range
    distance_robust_range = plane_distances[
        (plane_distances >= np.percentile(plane_distances, low_pct)) &
        (plane_distances <= np.percentile(plane_distances, high_pct))
    ]
    
    if len(distance_robust_range) > 10:
        robust_mean = np.mean(distance_robust_range)
        robust_std = np.std(distance_robust_range)
    else:
        robust_mean = np.mean(plane_distances)
        robust_std = np.std(plane_distances)
    
    # Define acceptable distance range
    max_distance = robust_mean + distance_std_ratio * robust_std
    
    # Create inlier mask
    inlier_mask = plane_distances <= max_distance
    inlier_points = points[inlier_mask]
    
    print(f"  Plane-based filtering: max distance {max_distance:.3f} (robust mean±{distance_std_ratio}σ)")
    print(f"  Plane normal: [{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]")
    
    return inlier_points, inlier_mask


def _remove_z_elevation_outliers(points, z_std_ratio=2.5, percentile_range=(5, 95)):
    """
    Traditional Z-coordinate based elevation outlier removal.
    """
    n_points = points.shape[0]
    z_coords = points[:, 2]
    
    # Use robust statistics (percentiles) instead of mean/std to handle multi-level buildings
    z_low, z_high = np.percentile(z_coords, percentile_range)
    z_robust_range = z_coords[(z_coords >= z_low) & (z_coords <= z_high)]
    
    if len(z_robust_range) > 10:  # Enough points for robust statistics
        z_robust_mean = np.mean(z_robust_range)
        z_robust_std = np.std(z_robust_range)
    else:
        # Fallback to global statistics
        z_robust_mean = np.mean(z_coords)
        z_robust_std = np.std(z_coords)
    
    # Define acceptable Z range
    z_min = z_robust_mean - z_std_ratio * z_robust_std
    z_max = z_robust_mean + z_std_ratio * z_robust_std
    
    # Create inlier mask
    inlier_mask = (z_coords >= z_min) & (z_coords <= z_max)
    inlier_points = points[inlier_mask]
    
    print(f"Elevation outlier removal: {np.sum(~inlier_mask)}/{n_points} points removed ({100*np.sum(~inlier_mask)/n_points:.1f}%)")
    print(f"  Acceptable Z range: [{z_min:.3f}, {z_max:.3f}] (robust mean±{z_std_ratio}σ)")
    
    return inlier_points, inlier_mask


def remove_density_outliers(points, density_threshold=0.5, radius=0.1, auto_scale_radius_param=True):
    """
    Remove density outliers - points in regions with abnormally low point density.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        density_threshold (float): Minimum relative density (0-1) required
        radius (float): Radius for density computation (auto-scaled if enabled)
        auto_scale_radius_param (bool): Whether to auto-scale radius based on point density
        
    Returns:
        tuple: (inlier_points, inlier_mask) where inlier_mask is boolean array
    """
    n_points = points.shape[0]
    
    # Handle edge cases with very small point clouds
    if n_points < 2:
        return points.copy(), np.ones(n_points, dtype=bool)
    
    # Auto-scale radius if requested
    if auto_scale_radius_param:
        original_radius = radius
        radius = auto_scale_radius(points, base_radius=radius)
        if abs(radius - original_radius) > 0.001:
            print(f"  Auto-scaled radius: {original_radius:.4f} → {radius:.4f}")
    
    # Compute local density for each point
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(points)
    
    # Vectorized radius neighbor search - much faster than Python loop
    neighbor_indices = nbrs.radius_neighbors(points, radius=radius, return_distance=False)
    
    # Count neighbors for each point (excluding self)
    densities = np.fromiter((len(indices) - 1 for indices in neighbor_indices), 
                           dtype=np.float32, count=n_points)
    
    # Use percentile-based thresholding instead of max-based (more robust)
    # Keep points in the top (1 - density_threshold) percentile of density
    percentile_threshold = (1.0 - density_threshold) * 100
    min_density_required = np.percentile(densities, percentile_threshold)
    
    # Create inlier mask
    inlier_mask = densities >= min_density_required
    inlier_points = points[inlier_mask]
    
    print(f"Density outlier removal: {np.sum(~inlier_mask)}/{n_points} points removed ({100*np.sum(~inlier_mask)/n_points:.1f}%)")
    print(f"  Density threshold: {min_density_required:.1f} neighbors (top {100-percentile_threshold:.0f}% densest regions)")
    
    return inlier_points, inlier_mask


def clean_point_cloud(points, remove_statistical=True, remove_radius=True, remove_elevation=True,
                     remove_density=False, stat_k=20, stat_std_ratio=2.0, radius_threshold=0.05, 
                     radius_min_neighbors=2, elev_std_ratio=2.5, elev_percentiles=(5, 95),
                     elev_use_plane=False, density_threshold=0.5, density_radius=0.1,
                     auto_scale_radii=True, neighbor_cache=None):
    """
    Comprehensive point cloud cleaning with multiple outlier removal methods.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        remove_statistical (bool): Apply statistical outlier removal
        remove_radius (bool): Apply radius outlier removal  
        remove_elevation (bool): Apply elevation outlier removal
        remove_density (bool): Apply density-based outlier removal
        stat_k (int): K-neighbors for statistical outlier detection
        stat_std_ratio (float): Standard deviation ratio for statistical outliers
        radius_threshold (float): Radius for radius outlier detection
        radius_min_neighbors (int): Minimum neighbors for radius outlier detection
        elev_std_ratio (float): Standard deviation ratio for elevation outliers
        elev_percentiles (tuple): Percentile range for robust elevation statistics
        elev_use_plane (bool): Use PCA-based plane detection for elevation filtering
        density_threshold (float): Minimum relative density for density outlier removal
        density_radius (float): Radius for density computation
        auto_scale_radii (bool): Auto-scale radius parameters based on point density
        neighbor_cache (NeighborCache): Optional cache for neighbor trees
        
    Returns:
        tuple: (cleaned_points, combined_mask) where combined_mask indicates which original points remain
    """
    current_points = points.copy()
    n_original = len(points)
    combined_mask = np.ones(n_original, dtype=bool)
    
    # Create neighbor cache if not provided
    if neighbor_cache is None:
        neighbor_cache = NeighborCache()
    
    print(f"\n=== Point Cloud Cleaning (starting with {n_original} points) ===")
    
    # 1. Statistical outlier removal
    if remove_statistical:
        clean_points, stat_mask = remove_statistical_outliers(
            current_points, k_neighbors=stat_k, std_ratio=stat_std_ratio
        )
        
        # Update combined mask
        temp_mask = np.zeros(n_original, dtype=bool)
        temp_mask[combined_mask] = stat_mask
        combined_mask = temp_mask
        current_points = clean_points
    
    # 2. Radius outlier removal
    if remove_radius:
        clean_points, radius_mask = remove_radius_outliers(
            current_points, radius=radius_threshold, min_neighbors=radius_min_neighbors,
            auto_scale_radius_param=auto_scale_radii
        )
        
        # Update combined mask
        temp_mask = np.zeros(n_original, dtype=bool)
        current_indices = np.where(combined_mask)[0]
        temp_mask[current_indices[radius_mask]] = True
        combined_mask = temp_mask
        current_points = clean_points
    
    # 3. Elevation outlier removal  
    if remove_elevation and len(current_points) > 0:
        clean_points, elev_mask = remove_elevation_outliers(
            current_points, z_std_ratio=elev_std_ratio, percentile_range=elev_percentiles,
            use_dominant_plane=elev_use_plane
        )
        
        # Update combined mask
        temp_mask = np.zeros(n_original, dtype=bool)
        current_indices = np.where(combined_mask)[0]
        temp_mask[current_indices[elev_mask]] = True
        combined_mask = temp_mask
        current_points = clean_points
    elif remove_elevation and len(current_points) == 0:
        print("Elevation outlier removal: Skipped - no points remaining after previous steps")
    
    # 4. Density outlier removal (optional)
    if remove_density and len(current_points) > 0:
        clean_points, density_mask = remove_density_outliers(
            current_points, density_threshold=density_threshold, radius=density_radius,
            auto_scale_radius_param=auto_scale_radii
        )
        
        # Update combined mask
        temp_mask = np.zeros(n_original, dtype=bool)
        current_indices = np.where(combined_mask)[0]
        temp_mask[current_indices[density_mask]] = True
        combined_mask = temp_mask
        current_points = clean_points
    elif remove_density and len(current_points) == 0:
        print("Density outlier removal: Skipped - no points remaining after previous steps")
    
    n_final = len(current_points)
    total_removed = n_original - n_final
    
    print(f"=== Cleaning Summary: {total_removed}/{n_original} total points removed ({100*total_removed/n_original:.1f}%) ===")
    print(f"Final clean point cloud: {n_final} points\n")
    
    return current_points, combined_mask


def validate_point_cloud_quality(points, verbose=True):
    """
    Analyze point cloud quality and provide statistics.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        verbose (bool): Whether to print detailed statistics
        
    Returns:
        dict: Quality metrics and statistics
    """
    n_points = len(points)
    
    # Compute basic statistics
    means = np.mean(points, axis=0)
    stds = np.std(points, axis=0)
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    ranges = maxs - mins
    
    # Compute density statistics
    nbrs = NearestNeighbors(n_neighbors=min(10, n_points), algorithm='kd_tree').fit(points)
    distances, _ = nbrs.kneighbors(points)
    mean_nn_distance = np.mean(distances[:, 1])  # Mean distance to nearest neighbor
    
    # Aspect ratio analysis
    aspect_ratios = ranges / np.max(ranges)
    
    quality_metrics = {
        'n_points': n_points,
        'centroid': means,
        'std_dev': stds,
        'bounding_box': {'min': mins, 'max': maxs, 'range': ranges},
        'mean_nearest_neighbor_distance': mean_nn_distance,
        'aspect_ratios': aspect_ratios,
        'is_roughly_cubic': np.all(aspect_ratios > 0.1),  # No dimension is too thin
        'point_density': n_points / np.prod(ranges) if np.prod(ranges) > 0 else 0
    }
    
    if verbose:
        print(f"\n=== Point Cloud Quality Analysis ===")
        print(f"Number of points: {n_points}")
        print(f"Centroid: ({means[0]:.3f}, {means[1]:.3f}, {means[2]:.3f})")
        print(f"Bounding box: X[{mins[0]:.3f}, {maxs[0]:.3f}], Y[{mins[1]:.3f}, {maxs[1]:.3f}], Z[{mins[2]:.3f}, {maxs[2]:.3f}]")
        print(f"Dimensions: {ranges[0]:.3f} × {ranges[1]:.3f} × {ranges[2]:.3f}")
        print(f"Mean nearest neighbor distance: {mean_nn_distance:.4f}")
        print(f"Point density: {quality_metrics['point_density']:.1f} points/unit³")
        print(f"Aspect ratios (X:Y:Z): {aspect_ratios[0]:.2f}:{aspect_ratios[1]:.2f}:{aspect_ratios[2]:.2f}")
        
        if quality_metrics['is_roughly_cubic']:
            print("Good aspect ratios - no overly thin dimensions")
        else:
            print("Warning: Point cloud has very thin dimensions")
    
    return quality_metrics