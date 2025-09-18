#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-15
# @Author  : GitHub Copilot
# @File    : preprocess.py
# @Purpose : Comprehensive preprocessing pipeline for 3D building point clouds

import numpy as np
import yaml
import os
from .normalization import normalize_data
from .outlier_removal import clean_point_cloud
from .surface_grouping import surface_aware_normalization_pipeline
from .weight_assignment import compute_border_weights


class Building3DPreprocessor:
    """
    Comprehensive preprocessing pipeline for 3D building point clouds.
    Integrates normalization, outlier removal, surface grouping, and border weight computation.
    """
    
    def __init__(self, config=None):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with preprocessing parameters
        """
        self.config = config or {}
        
        # Default configuration
        self.default_config = {
            'normalize': True,
            'use_outlier_removal': True,
            'use_surface_grouping': False,
            'compute_border_weights': True,
            'outlier_params': {
                'stat_k': 15,
                'stat_std_ratio': 2.5,
                'radius_threshold': 0.04,
                'radius_min_neighbors': 3,
                'elev_std_ratio': 2.5,
                'auto_scale_radii': True
            },
            'grouping_params': {
                'coarse_params': {
                    'eps': 0.05,
                    'min_samples': 5,
                    'auto_scale_eps': True
                },
                'refinement_params': {
                    'merge_distance': 0.05,
                    'min_group_size': 15
                }
            },
            'weight_params': {
                'k_neighbors': 25,
                'normalize_weights': True,
                'multi_scale': False,
                'use_normal_analysis': False  # Disabled since normals are removed
            }
        }
        
        # Merge user config with defaults
        self._merge_config()
    
    def _merge_config(self):
        """Merge user configuration with defaults."""
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
            elif isinstance(value, dict) and isinstance(self.config[key], dict):
                # Deep merge for nested dictionaries
                merged_dict = value.copy()
                merged_dict.update(self.config[key])
                self.config[key] = merged_dict
    
    def process_point_cloud(self, point_cloud, wireframe_vertices=None, verbose=True):
        """
        Process a point cloud through the complete pipeline.
        
        Args:
            point_cloud (np.ndarray): Input point cloud with shape (N, D) where D >= 3
            wireframe_vertices (np.ndarray, optional): Wireframe vertices to transform consistently
            verbose (bool): Whether to print processing information
            
        Returns:
            dict: Dictionary containing processed data and metadata
        """
        if verbose:
            print(f"\n=== Building3D Preprocessing Pipeline ===")
            print(f"Input point cloud shape: {point_cloud.shape}")
            if wireframe_vertices is not None:
                print(f"Wireframe vertices shape: {wireframe_vertices.shape}")
        
        results = {
            'processed_points': point_cloud.copy(),
            'processed_wireframe': wireframe_vertices.copy() if wireframe_vertices is not None else None,
            'metadata': {
                'original_shape': point_cloud.shape,
                'processing_steps': []
            }
        }
        
        # Step 1: Normalization and basic processing
        if self.config['normalize']:
            if verbose:
                print("\n--- Step 1: Normalization ---")
            
            if self.config['use_surface_grouping']:
                results = self._apply_surface_aware_processing(results, verbose)
            else:
                results = self._apply_standard_normalization(results, verbose)
        
        # Step 2: Compute border weights
        if self.config['compute_border_weights']:
            if verbose:
                print("\n--- Step 2: Border Weight Computation ---")
            results = self._compute_border_weights(results, verbose)
        
        if verbose:
            print(f"\n=== Pipeline Complete ===")
            print(f"Final point cloud shape: {results['processed_points'].shape}")
            print(f"Processing steps: {', '.join(results['metadata']['processing_steps'])}")
        
        return results
    
    def _apply_surface_aware_processing(self, results, verbose):
        """Apply surface-aware normalization pipeline."""
        point_cloud = results['processed_points']
        wireframe_vertices = results['processed_wireframe']
        
        if verbose:
            print("Using surface-aware processing pipeline")
        
        # Configure pipeline parameters
        outlier_params = self.config['outlier_params'].copy()
        grouping_params = self.config['grouping_params'].copy()
        
        processing_params = {
            'apply_outlier_removal': self.config['use_outlier_removal'],
            'compute_normals': False,  # Disabled - normals removed from pipeline
            'use_neighbor_cache': True
        }
        
        # Run surface-aware pipeline
        surface_results = surface_aware_normalization_pipeline(
            point_cloud[:, 0:3],  # Only XYZ coordinates for grouping
            outlier_removal_params=outlier_params,
            grouping_params=grouping_params,
            processing_params=processing_params
        )
        
        if surface_results is not None:
            # Integrate results back into point cloud
            cleaned_points = surface_results['cleaned_points']
            group_ids = surface_results.get('group_ids', None)
            
            # Normalize coordinates to unit scale
            centroid = np.mean(cleaned_points, axis=0)
            centered_points = cleaned_points - centroid
            max_distance = np.max(np.linalg.norm(centered_points, axis=1))
            
            if max_distance > 1e-6:
                normalized_coords = centered_points / max_distance
            else:
                normalized_coords = centered_points
            
            # Apply same transformation to wireframe if provided
            if wireframe_vertices is not None:
                wireframe_vertices = (wireframe_vertices - centroid) / max_distance
                results['processed_wireframe'] = wireframe_vertices
            
            # Reconstruct full point cloud with surface information (no normals)
            n_cleaned = len(cleaned_points)
            
            if point_cloud.shape[1] >= 3:
                # Determine output dimensions
                output_dims = 3  # XYZ
                if group_ids is not None:
                    output_dims += 1  # +1 for group ID
                if point_cloud.shape[1] > 3:
                    output_dims += (point_cloud.shape[1] - 3)  # preserve other features
                
                reconstructed_pc = np.zeros((n_cleaned, output_dims))
                
                # Set coordinates
                reconstructed_pc[:, 0:3] = normalized_coords
                col_idx = 3
                
                # Add group IDs if available
                if group_ids is not None:
                    reconstructed_pc[:, col_idx] = group_ids
                    col_idx += 1
                
                # Preserve original features if any (colors, intensity, etc.)
                if point_cloud.shape[1] > 3:
                    original_features = point_cloud.shape[1] - 3
                    if col_idx + original_features <= output_dims:
                        # Map features from original to cleaned points using nearest neighbors
                        from sklearn.neighbors import NearestNeighbors
                        nbrs = NearestNeighbors(n_neighbors=1).fit(point_cloud[:, 0:3])
                        distances, indices = nbrs.kneighbors(cleaned_points)
                        reconstructed_pc[:, col_idx:col_idx+original_features] = point_cloud[indices.flatten(), 3:3+original_features]
                
                results['processed_points'] = reconstructed_pc
            else:
                # Simple case: just coordinates + group ids
                components = [normalized_coords]
                if group_ids is not None:
                    components.append(group_ids.reshape(-1, 1))
                
                results['processed_points'] = np.column_stack(components)
            
            # Store metadata
            results['metadata']['centroid'] = centroid
            results['metadata']['max_distance'] = max_distance
            results['metadata']['has_normals'] = False  # Normals removed from pipeline
            results['metadata']['has_group_ids'] = group_ids is not None
            results['metadata']['processing_steps'].append('surface_aware_normalization')
            
            if verbose:
                print(f"Surface-aware processing successful")
                print(f"Points: {len(point_cloud)} -> {n_cleaned}")
                print(f"Features: groups={group_ids is not None}")
        
        else:
            if verbose:
                print("Surface-aware processing failed, falling back to standard normalization")
            results = self._apply_standard_normalization(results, verbose)
        
        return results
    
    def _apply_standard_normalization(self, results, verbose):
        """Apply standard normalization pipeline."""
        point_cloud = results['processed_points']
        wireframe_vertices = results['processed_wireframe']
        
        if verbose:
            print("Using standard normalization pipeline")
        
        # Apply normalization with outlier removal
        processed_pc, processed_wf, centroid, max_distance = normalize_data(
            point_cloud, 
            wireframe_vertices, 
            clean_outliers=self.config['use_outlier_removal'],
            outlier_params=self.config['outlier_params']
        )
        
        results['processed_points'] = processed_pc
        results['processed_wireframe'] = processed_wf
        results['metadata']['centroid'] = centroid
        results['metadata']['max_distance'] = max_distance
        results['metadata']['has_normals'] = False  # Normals removed from pipeline
        results['metadata']['processing_steps'].append('standard_normalization')
        
        if verbose:
            print(f"Standard normalization complete")
            print(f"Points: {len(point_cloud)} -> {len(processed_pc)}")
        
        return results
    
    def _compute_border_weights(self, results, verbose):
        """Compute border weights for the processed point cloud."""
        point_cloud = results['processed_points']
        
        # Extract coordinates only (no normals available)
        coordinates = point_cloud[:, 0:3]
        
        # Compute border weights without normal analysis
        weight_params = self.config['weight_params']
        border_weights = compute_border_weights(
            coordinates,
            k_neighbors=weight_params['k_neighbors'],
            normalize_weights=weight_params['normalize_weights'],
            multi_scale=weight_params['multi_scale']
        )
        
        # Add border weights to the point cloud
        results['processed_points'] = np.column_stack([point_cloud, border_weights])
        results['metadata']['has_border_weights'] = True
        results['metadata']['border_weight_column'] = point_cloud.shape[1]  # Index of border weight column
        results['metadata']['processing_steps'].append('border_weights')
        
        if verbose:
            print(f"Border weights computed and added to point cloud")
            print(f"Border weight statistics:")
            print(f"  Mean: {np.mean(border_weights):.4f}")
            print(f"  Std: {np.std(border_weights):.4f}")
            print(f"  Min/Max: {np.min(border_weights):.4f}/{np.max(border_weights):.4f}")
        
        return results
    
    def process_dataset_sample(self, dataset_sample, verbose=True):
        """
        Process a sample from the Building3DReconstructionDataset.
        
        Args:
            dataset_sample (dict): Sample from dataset with 'point_clouds', 'wf_vertices', etc.
            verbose (bool): Whether to print processing information
            
        Returns:
            dict: Updated dataset sample with processed data
        """
        point_cloud = dataset_sample['point_clouds']
        wireframe_vertices = dataset_sample.get('wf_vertices', None)
        
        # Process through pipeline
        results = self.process_point_cloud(point_cloud, wireframe_vertices, verbose)
        
        # Update dataset sample
        updated_sample = dataset_sample.copy()
        updated_sample['point_clouds'] = results['processed_points']
        if wireframe_vertices is not None:
            updated_sample['wf_vertices'] = results['processed_wireframe']
        
        # Add processing metadata
        updated_sample['preprocessing_metadata'] = results['metadata']
        
        return updated_sample


def create_default_preprocessor(config_file=None):
    """Create a preprocessor using configuration from YAML file."""
    if config_file is None:
        # Default path relative to the utils directory
        config_file = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset_config.yaml')
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Extract preprocessor config from Building3D section
    preprocessor_config = yaml_config['Building3D']['preprocessor']
        
    return Building3DPreprocessor(preprocessor_config)