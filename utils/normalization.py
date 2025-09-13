#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11
# @Author  : extracted from building3d.py
# @File    : normalization.py
# @Purpose : Data normalization utilities for 3D point clouds and wireframes

import numpy as np


def normalize_data(point_cloud, wf_vertices):
    """
    Normalize point cloud and wireframe data to be centered at origin with unit scale.
    
    Args:
        point_cloud (np.ndarray): Point cloud data with shape (N, 8) where first 3 columns are XYZ
        wf_vertices (np.ndarray): Wireframe vertices with shape (M, 3) 
        
    Returns:
        tuple: (normalized_point_cloud, normalized_wf_vertices, centroid, max_distance)
            - normalized_point_cloud: Point cloud centered and scaled
            - normalized_wf_vertices: Wireframe vertices centered and scaled  
            - centroid: Original centroid of the point cloud (for denormalization)
            - max_distance: Original max distance from centroid (for denormalization)
    """
    # Calculate centroid from point cloud XYZ coordinates
    centroid = np.mean(point_cloud[:, 0:3], axis=0)
    
    # Center the data by subtracting centroid
    normalized_point_cloud = point_cloud.copy()
    normalized_point_cloud[:, 0:3] -= centroid
    
    # Calculate max distance for scaling
    max_distance = np.max(np.linalg.norm(normalized_point_cloud[:, 0:3], axis=1))
    
    # Scale to unit size
    normalized_point_cloud[:, 0:3] /= max_distance
    
    # Apply same transformation to wireframe vertices
    normalized_wf_vertices = wf_vertices.copy()
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
        point_cloud (np.ndarray): Point cloud data with shape (N, 8) where first 3 columns are XYZ
        
    Returns:
        tuple: (centroid, max_distance)
            - centroid: Centroid of the point cloud
            - max_distance: Maximum distance from centroid
    """
    centroid = np.mean(point_cloud[:, 0:3], axis=0)
    centered_points = point_cloud[:, 0:3] - centroid
    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
    
    return centroid, max_distance