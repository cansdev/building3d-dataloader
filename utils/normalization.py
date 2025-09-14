#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11
# @Author  : extracted from building3d.py
# @File    : normalization.py
# @Purpose : Data normalization utilities for 3D point clouds and wireframes

import numpy as np
from sklearn.neighbors import NearestNeighbors


def estimate_normals(points, k_neighbors=10, flip_normals=True):
    """
    Estimate surface normals for point cloud using PCA on local neighborhoods.
    
    Args:
        points (np.ndarray): Point cloud coordinates with shape (N, 3)
        k_neighbors (int): Number of neighbors to use for normal estimation
        flip_normals (bool): Whether to flip normals towards consistent orientation
        
    Returns:
        np.ndarray: Estimated normals with shape (N, 3)
    """
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


def normalize_data(point_cloud, wf_vertices, compute_normals=False, k_neighbors=10):
    """
    Normalize point cloud and wireframe data to be centered at origin with unit scale.
    Optionally compute surface normals.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, 8) where first 3 columns are XYZ
        wf_vertices (np.ndarray): Wireframe vertices with shape (M, 3) 
        compute_normals (bool): Whether to compute and append surface normals
        k_neighbors (int): Number of neighbors for normal estimation
        
    Returns:
        tuple: (normalized_point_cloud, normalized_wf_vertices, centroid, max_distance, normals)
            - normalized_point_cloud: Point cloud centered and scaled, optionally with normals
            - normalized_wf_vertices: Wireframe vertices centered and scaled  
            - centroid: Original centroid of the point cloud (for denormalization)
            - max_distance: Original max distance from centroid (for denormalization)
            - normals: Estimated surface normals (None if compute_normals=False)
    """
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
        normals = estimate_normals(normalized_point_cloud[:, 0:3], k_neighbors=k_neighbors)
        
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