#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11
# @Author  : extracted from building3d.py
# @File    : normalization.py
# @Purpose : Data normalization utilities for 3D point clouds and wireframes

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .outlier_removal import clean_point_cloud


def compute_local_density(points, radius=0.1):
    """
    Compute local point density for adaptive k-neighbor selection.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        radius (float): Radius for density computation
        
    Returns:
        np.ndarray: Local density for each point
    """
    n_points = points.shape[0]
    densities = np.zeros(n_points)
    
    # Use radius-based neighbors for density estimation
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(points)
    
    for i in range(n_points):
        # Find all neighbors within radius
        indices = nbrs.radius_neighbors([points[i]], radius=radius, return_distance=False)[0]
        densities[i] = len(indices) - 1  # Exclude the point itself
    
    return densities


def adaptive_k_neighbors(points, base_k=10, min_k=5, max_k=30, density_radius=0.1):
    """
    Compute adaptive k for each point based on local density.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        base_k (int): Base number of neighbors
        min_k (int): Minimum k value
        max_k (int): Maximum k value
        density_radius (float): Radius for density computation
        
    Returns:
        np.ndarray: Adaptive k values for each point
    """
    densities = compute_local_density(points, density_radius)
    
    # Normalize densities to [0, 1] range
    min_density = np.min(densities)
    max_density = np.max(densities)
    
    if max_density > min_density:
        normalized_densities = (densities - min_density) / (max_density - min_density)
    else:
        normalized_densities = np.ones_like(densities) * 0.5
    
    # Inverse relationship: low density → high k, high density → low k
    k_values = base_k + (max_k - base_k) * (1 - normalized_densities)
    k_values = np.clip(k_values, min_k, max_k).astype(int)
    
    return k_values


def estimate_normals_robust(points, base_k_neighbors=10, curvature_threshold=0.1, 
                           smooth_iterations=1, smooth_radius=0.05, adaptive_k=True):
    """
    Enhanced surface normal estimation with planarity filtering, adaptive k, and smoothing.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        base_k_neighbors (int): Base number of neighbors for normal estimation
        curvature_threshold (float): Threshold for planarity filtering (λ₃/(λ₁+λ₂+λ₃))
        smooth_iterations (int): Number of smoothing iterations
        smooth_radius (float): Radius for normal smoothing
        adaptive_k (bool): Whether to use adaptive k-neighbor selection
        
    Returns:
        tuple: (normals, quality_scores) where quality_scores indicate normal reliability
    """
    n_points = points.shape[0]
    normals = np.zeros_like(points)
    quality_scores = np.zeros(n_points)
    
    # 🟡 B. Adaptive k-neighbor selection
    if adaptive_k:
        k_values = adaptive_k_neighbors(points, base_k=base_k_neighbors)
        print(f"Using adaptive k: min={np.min(k_values)}, max={np.max(k_values)}, mean={np.mean(k_values):.1f}")
    else:
        k_values = np.full(n_points, base_k_neighbors)
    
    # Find maximum k for neighborhood computation
    max_k = min(np.max(k_values), n_points - 1)  # Ensure max_k < n_points
    if max_k < 5:
        print(f"Warning: Very few points ({n_points}), using simplified normal estimation")
        max_k = min(5, n_points - 1)
        k_values = np.minimum(k_values, max_k)
    
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='kd_tree').fit(points)
    _, all_indices = nbrs.kneighbors(points)
    
    for i in range(n_points):
        k = k_values[i]
        
        # Get local neighborhood
        neighborhood_indices = all_indices[i, :k]
        neighborhood = points[neighborhood_indices]
        
        # Center the neighborhood
        centered = neighborhood - np.mean(neighborhood, axis=0)
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered.T)
        
        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvalues in descending order: λ₁ ≥ λ₂ ≥ λ₃
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # 🟢 A. Planarity filtering using curvature measure
        lambda1, lambda2, lambda3 = eigenvalues
        total_curvature = lambda1 + lambda2 + lambda3
        
        if total_curvature > 1e-8:  # Avoid division by zero
            curvature_ratio = lambda3 / total_curvature
            
            if curvature_ratio > curvature_threshold:
                # High curvature - not planar, lower quality
                quality_scores[i] = 1.0 - curvature_ratio  # Lower is worse
            else:
                # Low curvature - planar surface, high quality
                quality_scores[i] = 1.0  # High quality for planar surfaces
        else:
            # Degenerate case
            quality_scores[i] = 0.0
        
        # Normal is eigenvector with smallest eigenvalue (λ₃)
        normal = eigenvectors[:, 2]  # Third column (smallest eigenvalue)
        normals[i] = normal
    
    # Orient normals consistently
    normals = orient_normals_consistently(points, normals)
    
    # Normalize to unit vectors
    normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
    
    # 🟣 C. Smooth normals with neighbors
    if smooth_iterations > 0:
        normals = smooth_normals_with_neighbors(points, normals, quality_scores, 
                                               smooth_radius, smooth_iterations)
    
    return normals, quality_scores


def smooth_normals_with_neighbors(points, normals, quality_scores, radius=0.05, iterations=1):
    """
    Smooth normals by weighted averaging with neighbors.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        normals (np.ndarray): Surface normals with shape (N, 3)
        quality_scores (np.ndarray): Quality scores for weighting
        radius (float): Radius for neighbor search
        iterations (int): Number of smoothing iterations
        
    Returns:
        np.ndarray: Smoothed normals
    """
    n_points = points.shape[0]
    smoothed_normals = normals.copy()
    
    # Build neighbor structure
    nbrs = NearestNeighbors(algorithm='kd_tree').fit(points)
    
    for iteration in range(iterations):
        new_normals = smoothed_normals.copy()
        
        for i in range(n_points):
            # Find neighbors within radius
            neighbor_indices = nbrs.radius_neighbors([points[i]], radius=radius, return_distance=False)[0]
            
            if len(neighbor_indices) > 1:  # Exclude only the point itself
                # Get neighbor normals and distances
                neighbor_points = points[neighbor_indices]
                neighbor_normals = smoothed_normals[neighbor_indices]
                neighbor_qualities = quality_scores[neighbor_indices]
                
                # Compute weights based on distance and quality
                distances = np.linalg.norm(neighbor_points - points[i], axis=1)
                distance_weights = np.exp(-distances / (radius * 0.5))  # Gaussian falloff
                
                # Combine distance and quality weights
                total_weights = distance_weights * neighbor_qualities
                total_weights = total_weights / (np.sum(total_weights) + 1e-8)
                
                # Weighted average of normals
                weighted_normal = np.sum(neighbor_normals * total_weights[:, np.newaxis], axis=0)
                
                # Renormalize
                norm = np.linalg.norm(weighted_normal)
                if norm > 1e-8:
                    new_normals[i] = weighted_normal / norm
        
        smoothed_normals = new_normals
        
    print(f"Applied {iterations} iterations of normal smoothing with radius {radius}")
    return smoothed_normals


def estimate_normals(points, k_neighbors=10, flip_normals=True, robust=True, **kwargs):
    """
    Estimate surface normals for point cloud using PCA on local neighborhoods.
    Now supports robust estimation with planarity filtering, adaptive k, and smoothing.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        k_neighbors (int): Base number of neighbors for normal estimation
        flip_normals (bool): Whether to flip normals towards consistent orientation
        robust (bool): Whether to use robust estimation with enhancements
        **kwargs: Additional arguments for robust estimation
        
    Returns:
        np.ndarray: Estimated normals with shape (N, 3)
    """
    if robust:
        # Use enhanced robust estimation
        normals, quality_scores = estimate_normals_robust(
            points, 
            base_k_neighbors=k_neighbors,
            curvature_threshold=kwargs.get('curvature_threshold', 0.1),
            smooth_iterations=kwargs.get('smooth_iterations', 1),
            smooth_radius=kwargs.get('smooth_radius', 0.05),
            adaptive_k=kwargs.get('adaptive_k', True)
        )
        return normals
    else:
        # Use original simple estimation
        n_points = points.shape[0]
        normals = np.zeros_like(points)
        
        # Find k-nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='kd_tree').fit(points)
        _, indices = nbrs.kneighbors(points)
        
        for i in range(n_points):
            # Get local neighborhood
            neighborhood = points[indices[i]]
            
            # Center the neighborhood
            centered = neighborhood - np.mean(neighborhood, axis=0)
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered.T)
            
            # Find eigenvector corresponding to smallest eigenvalue (normal direction)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            normal = eigenvectors[:, 0]  # Smallest eigenvalue eigenvector
            
            normals[i] = normal
        
        # Optional: flip normals for consistent orientation
        if flip_normals:
            normals = orient_normals_consistently(points, normals)
        
        # Normalize to unit vectors
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        return normals


def orient_normals_consistently(points, normals):
    """
    Orient normals consistently using a simple heuristic.
    Points normals away from the centroid of the point cloud.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        normals (np.ndarray): Estimated normals with shape (N, 3)
        
    Returns:
        np.ndarray: Consistently oriented normals
    """
    centroid = np.mean(points, axis=0)
    
    # Vector from centroid to each point
    to_point = points - centroid
    
    # If normal points towards centroid, flip it
    dot_product = np.sum(normals * to_point, axis=1)
    flip_mask = dot_product < 0
    
    normals[flip_mask] *= -1
    
    return normals


def normalize_data(point_cloud, wf_vertices, compute_normals=False, k_neighbors=10, 
                  clean_outliers=True, outlier_params=None):
    """
    Normalize point cloud and wireframe data to be centered at origin with unit scale.
    Optionally compute surface normals with outlier removal for better quality.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, 8) where first 3 columns are XYZ
        wf_vertices (np.ndarray): Wireframe vertices with shape (M, 3) 
        compute_normals (bool): Whether to compute and append surface normals
        k_neighbors (int): Number of neighbors for normal estimation
        clean_outliers (bool): Whether to remove outliers before normal computation
        outlier_params (dict): Parameters for outlier removal (optional)
        
    Returns:
        tuple: (normalized_point_cloud, normalized_wf_vertices, centroid, max_distance, normals)
            - normalized_point_cloud: Point cloud centered and scaled, optionally with normals
            - normalized_wf_vertices: Wireframe vertices centered and scaled  
            - centroid: Original centroid of the point cloud (for denormalization)
            - max_distance: Original max distance from centroid (for denormalization)
            - normals: Estimated surface normals (None if compute_normals=False)
    """
    # Default outlier removal parameters
    if outlier_params is None:
        outlier_params = {
            'stat_k': 20,
            'stat_std_ratio': 2.0,
            'radius_threshold': 0.05,
            'radius_min_neighbors': 2,
            'elev_std_ratio': 2.5,
            'elev_percentiles': (5, 95)
        }
    
    # Store original point cloud for outlier removal
    original_coords = point_cloud[:, 0:3].copy()
    
    # Calculate centroid from point cloud XYZ coordinates
    centroid = np.mean(point_cloud[:, 0:3], axis=0)
    
    # Center the data by subtracting centroid
    normalized_point_cloud = point_cloud.copy().astype(np.float64)
    normalized_point_cloud[:, 0:3] -= centroid
    
    # Calculate max distance for scaling
    max_distance = np.max(np.linalg.norm(normalized_point_cloud[:, 0:3], axis=1))
    
    # Scale to unit size
    normalized_point_cloud[:, 0:3] /= max_distance
    
    # Apply same transformation to wireframe vertices
    normalized_wf_vertices = wf_vertices.copy().astype(np.float64)
    normalized_wf_vertices -= centroid
    normalized_wf_vertices /= max_distance
    
    # Compute normals if requested
    normals = None
    if compute_normals:
        print("Computing surface normals...")
        
        # Clean point cloud coordinates before normal computation
        if clean_outliers:
            # Apply cleaning to normalized coordinates for better scale-invariant results
            clean_coords, clean_mask = clean_point_cloud(
                normalized_point_cloud[:, 0:3],
                remove_statistical=True,
                remove_radius=True,
                remove_elevation=True,
                **outlier_params
            )
            
            # Compute normals only on clean points
            clean_normals = estimate_normals(clean_coords, k_neighbors=k_neighbors, robust=True)
            
            # Create full normal array, filling outlier positions with [0,0,1] (up direction)
            normals = np.zeros((normalized_point_cloud.shape[0], 3))
            normals[clean_mask] = clean_normals
            normals[~clean_mask] = [0, 0, 1]  # Default upward normal for outliers
            
            print(f"Computed normals: {np.sum(clean_mask)} clean points, {np.sum(~clean_mask)} outliers (filled with default normals)")
        else:
            # Compute normals on all points without cleaning
            normals = estimate_normals(normalized_point_cloud[:, 0:3], k_neighbors=k_neighbors, robust=True)
        
        # Append normals to point cloud data (columns 8, 9, 10)
        if normalized_point_cloud.shape[1] == 8:
            # Extend to accommodate normals
            extended_data = np.zeros((normalized_point_cloud.shape[0], 11))
            extended_data[:, :8] = normalized_point_cloud
            extended_data[:, 8:11] = normals
            normalized_point_cloud = extended_data
        else:
            # Replace or append normals
            if normalized_point_cloud.shape[1] >= 11:
                normalized_point_cloud[:, 8:11] = normals
            else:
                # Append normals
                normalized_point_cloud = np.column_stack([normalized_point_cloud, normals])
    
    return normalized_point_cloud, normalized_wf_vertices, centroid, max_distance, normals


def denormalize_data(normalized_point_cloud, normalized_wf_vertices, centroid, max_distance):
    """
    Reverse the normalization process to get back original coordinates.
    Note: Normals (if present) are preserved as they are direction vectors.
    
    Args:
        normalized_point_cloud (np.ndarray): Normalized point cloud data (may include normals)
        normalized_wf_vertices (np.ndarray): Normalized wireframe vertices
        centroid (np.ndarray): Original centroid used for normalization
        max_distance (float): Original max distance used for normalization
        
    Returns:
        tuple: (original_point_cloud, original_wf_vertices)
            - original_point_cloud: Point cloud in original coordinate system
            - original_wf_vertices: Wireframe vertices in original coordinate system
    """
    # Reverse scaling and centering for point cloud
    original_point_cloud = normalized_point_cloud.copy()
    original_point_cloud[:, 0:3] *= max_distance
    original_point_cloud[:, 0:3] += centroid
    
    # Note: Normals (columns 8:11 if present) remain unchanged as they are unit direction vectors
    
    # Reverse scaling and centering for wireframe vertices
    original_wf_vertices = normalized_wf_vertices.copy()
    original_wf_vertices *= max_distance
    original_wf_vertices += centroid
    
    return original_point_cloud, original_wf_vertices


def get_normalization_params(point_cloud):
    """
    Calculate normalization parameters without applying them.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, 8+) where first 3 columns are XYZ
        
    Returns:
        tuple: (centroid, max_distance)
            - centroid: Centroid of the point cloud
            - max_distance: Maximum distance from centroid
    """
    centroid = np.mean(point_cloud[:, 0:3], axis=0)
    centered_points = point_cloud[:, 0:3] - centroid
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    
    return centroid, max_distance


def extract_normals(point_cloud_with_normals):
    """
    Extract normals from point cloud data that includes normals.
    
    Args:
        point_cloud_with_normals (np.ndarray): Point cloud with normals in columns 8:11
        
    Returns:
        tuple: (point_cloud_without_normals, normals)
            - point_cloud_without_normals: Original point cloud data (N, 8)
            - normals: Surface normals (N, 3) or None if not present
    """
    if point_cloud_with_normals.shape[1] >= 11:
        point_cloud = point_cloud_with_normals[:, :8]
        normals = point_cloud_with_normals[:, 8:11]
        return point_cloud, normals
    else:
        return point_cloud_with_normals, None


def validate_normals(normals, tolerance=1e-3):
    """
    Validate that normals are unit vectors.
    
    Args:
        normals (np.ndarray): Normal vectors with shape (N, 3)
        tolerance (float): Tolerance for unit vector check
        
    Returns:
        bool: True if all normals are approximately unit vectors
    """
    if normals is None:
        return True
        
    norms = np.linalg.norm(normals, axis=1)
    return np.all(np.abs(norms - 1.0) < tolerance)