#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2025-09-15
# @Author  : Batuhan Arda Bekar
# @File    : preprocess.py
# @Purpose : Comprehensive preprocessing pipeline for 3D building point clouds

import numpy as np
import yaml
import os
import pickle
import hashlib
import time
from .normalization import normalize_data
from .outlier_removal import clean_point_cloud
from .surface_grouping import surface_aware_normalization_pipeline
from .weight_assignment import compute_border_weights


class Building3DPreprocessor:
    """
    Comprehensive preprocessing pipeline for 3D building point clouds.
    Integrates normalization, outlier removal, surface grouping, and border weight computation.
    """
    
    def __init__(self, config=None, cache_dir=None):
        """
        Initialize the preprocessor with configuration parameters.
        
        Args:
            config (dict): Configuration dictionary with preprocessing parameters
            cache_dir (str): Directory to store cached preprocessed data
        """
        self.config = config or {}
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'datasets', 'preprocessed')
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
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
    
    def _generate_cache_key(self, point_cloud_file, wireframe_file=None):
        """
        Generate a unique cache key based on input files and configuration.
        
        Args:
            point_cloud_file (str): Path to point cloud file
            wireframe_file (str): Path to wireframe file (optional)
            
        Returns:
            str: Unique cache key
        """
        # Get file modification times and sizes for cache invalidation
        pc_stat = os.stat(point_cloud_file)
        pc_info = f"{point_cloud_file}_{pc_stat.st_mtime}_{pc_stat.st_size}"
        
        wf_info = ""
        if wireframe_file and os.path.exists(wireframe_file):
            wf_stat = os.stat(wireframe_file)
            wf_info = f"_{wireframe_file}_{wf_stat.st_mtime}_{wf_stat.st_size}"
        
        # Include configuration in hash
        config_str = str(sorted(self.config.items()))
        
        # Create hash
        hash_input = f"{pc_info}{wf_info}_{config_str}".encode('utf-8')
        cache_key = hashlib.md5(hash_input).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key):
        """Get the full path for a cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _save_to_cache(self, cache_key, results):
        """
        Save preprocessing results to cache.
        
        Args:
            cache_key (str): Cache key
            results (dict): Preprocessing results to cache
        """
        try:
            cache_path = self._get_cache_path(cache_key)
            cache_data = {
                'results': results,
                'timestamp': time.time(),
                'config': self.config.copy()
            }
            
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            if hasattr(self, '_verbose') and self._verbose:
                print(f"Saved preprocessed data to cache: {cache_key}")
                
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_key}: {e}")
    
    def _load_from_cache(self, cache_key):
        """
        Load preprocessing results from cache.
        
        Args:
            cache_key (str): Cache key
            
        Returns:
            dict or None: Cached results if available and valid, None otherwise
        """
        try:
            cache_path = self._get_cache_path(cache_key)
            
            if not os.path.exists(cache_path):
                return None
            
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Verify cache validity (config match)
            if cache_data.get('config') != self.config:
                if hasattr(self, '_verbose') and self._verbose:
                    print(f"Cache invalid due to config change: {cache_key}")
                return None
            
            if hasattr(self, '_verbose') and self._verbose:
                cache_age = time.time() - cache_data.get('timestamp', 0)
                print(f"Loaded from cache (age: {cache_age:.1f}s): {cache_key}")
            
            return cache_data['results']
            
        except Exception as e:
            print(f"Warning: Failed to load cache {cache_key}: {e}")
            return None
    
    def clear_cache(self, max_age_days=30):
        """
        Clear old cache files.
        
        Args:
            max_age_days (int): Maximum age in days for cache files
        """
        try:
            current_time = time.time()
            max_age_seconds = max_age_days * 24 * 3600
            
            cleared_count = 0
            for cache_file in os.listdir(self.cache_dir):
                if cache_file.endswith('.pkl'):
                    cache_path = os.path.join(self.cache_dir, cache_file)
                    file_age = current_time - os.path.getmtime(cache_path)
                    
                    if file_age > max_age_seconds:
                        os.remove(cache_path)
                        cleared_count += 1
            
            if cleared_count > 0:
                print(f"Cleared {cleared_count} old cache files")
                
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")
    
    def process_point_cloud(self, point_cloud, wireframe_vertices=None, verbose=True, 
                           point_cloud_file=None, wireframe_file=None, use_cache=True):
        """
        Process a point cloud through the complete pipeline.
        
        Args:
            point_cloud (np.ndarray): Input point cloud with shape (N, D) where D >= 3
            wireframe_vertices (np.ndarray, optional): Wireframe vertices to transform consistently
            verbose (bool): Whether to print processing information
            point_cloud_file (str, optional): Path to point cloud file for caching
            wireframe_file (str, optional): Path to wireframe file for caching
            use_cache (bool): Whether to use caching
            
        Returns:
            dict: Dictionary containing processed data and metadata
        """
        self._verbose = verbose  # Store for cache methods
        
        # Try to load from cache if file paths are provided
        cache_key = None
        if use_cache and point_cloud_file:
            cache_key = self._generate_cache_key(point_cloud_file, wireframe_file)
            cached_results = self._load_from_cache(cache_key)
            if cached_results is not None:
                return cached_results
        
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
        
        # Save to cache if enabled and cache key available
        if use_cache and cache_key:
            self._save_to_cache(cache_key, results)
        
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
            
            # Reconstruct point cloud with X Y Z GroupID (BorderWeight to be added later)
            n_cleaned = len(cleaned_points)
            
            if point_cloud.shape[1] >= 8:  # Expecting X Y Z R G B A I format
                # Output: X Y Z GroupID (BorderWeight will be added in _compute_border_weights)
                output_dims = 4  # XYZ + GroupID (BorderWeight added later)
                reconstructed_pc = np.zeros((n_cleaned, output_dims))
                
                # Set coordinates
                reconstructed_pc[:, 0:3] = normalized_coords
                col_idx = 3
                
                # Add group IDs if available
                if group_ids is not None:
                    reconstructed_pc[:, col_idx] = group_ids
                    col_idx += 1
                
                # Note: BorderWeight will be added by _compute_border_weights step
                results['processed_points'] = reconstructed_pc
            else:
                # Fallback: just coordinates + group ids
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


def create_default_preprocessor(config_file=None, cache_dir=None):
    """Create a preprocessor using configuration from YAML file."""
    if config_file is None:
        # Default path relative to the utils directory
        config_file = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'dataset_config.yaml')
    
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Extract preprocessor config from Building3D section
    preprocessor_config = yaml_config['Building3D']['preprocessor']
    
    # Set default cache directory if not provided
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'preprocessed')
        
    return Building3DPreprocessor(preprocessor_config, cache_dir=cache_dir)