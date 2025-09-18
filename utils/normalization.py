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


def normalize_data(point_cloud, wf_vertices, clean_outliers=True, outlier_params=None):
    """
    Normalize point cloud and wireframe data to be centered at origin with unit scale.
    Applies outlier removal for better quality data.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, D) where first 3 columns are XYZ
        wf_vertices (np.ndarray): Wireframe vertices with shape (M, 3) 
        clean_outliers (bool): Whether to remove outliers before normalization
        outlier_params (dict): Parameters for outlier removal (optional)
        
    Returns:
        tuple: (normalized_point_cloud, normalized_wf_vertices, centroid, max_distance)
            - normalized_point_cloud: Point cloud centered and scaled without normals
            - normalized_wf_vertices: Wireframe vertices centered and scaled  
            - centroid: Original centroid of the point cloud (for denormalization)
            - max_distance: Original max distance from centroid (for denormalization)
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

    return normalized_point_cloud, normalized_wf_vertices, centroid, max_distance


def denormalize_data(normalized_point_cloud, normalized_wf_vertices, centroid, max_distance):
    """
    Reverse the normalization process to get back original coordinates.
    
    Args:
        normalized_point_cloud (np.ndarray): Normalized point cloud data
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
    
    # Reverse scaling and centering for wireframe vertices
    original_wf_vertices = normalized_wf_vertices.copy()
    original_wf_vertices *= max_distance
    original_wf_vertices += centroid
    
    return original_point_cloud, original_wf_vertices


def get_normalization_params(point_cloud):
    """
    Calculate normalization parameters without applying them.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, D+) where first 3 columns are XYZ
        
    Returns:
        tuple: (centroid, max_distance)
            - centroid: Centroid of the point cloud
            - max_distance: Maximum distance from centroid
    """
    centroid = np.mean(point_cloud[:, 0:3], axis=0)
    centered_points = point_cloud[:, 0:3] - centroid
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    
    return centroid, max_distance