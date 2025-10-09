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
            config (dict): Configuration dictionary with preprocessing parameters (required)
            cache_dir (str): Directory to store cached preprocessed data
        """
        if config is None:
            raise ValueError("Configuration is required. Please provide a config dictionary.")
        
        self.config = config
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '..', 'datasets', 'preprocessed')
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
    
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
                # Print summary even for cached data if verbose
                if verbose:
                    self._print_processing_summary(cached_results)
                return cached_results
        
        # Store source filename for summary
        source_filename = os.path.basename(point_cloud_file) if point_cloud_file else "unknown"
        
        results = {
            'processed_points': point_cloud.copy(),
            'processed_wireframe': wireframe_vertices.copy() if wireframe_vertices is not None else None,
            'metadata': {
                'original_shape': point_cloud.shape,
                'processing_steps': [],
                'source_file': source_filename
            }
        }
        
        # Step 1: Normalization and basic processing
        if self.config['normalize']:
            if self.config['use_surface_grouping']:
                results = self._apply_surface_aware_processing(results, verbose)
            else:
                results = self._apply_standard_normalization(results, verbose)
        
        # Step 2: Compute border weights
        if self.config['compute_border_weights']:
            results = self._compute_border_weights(results, verbose)
        
        # Final summary
        if verbose:
            self._print_processing_summary(results)
        
        # Save to cache if enabled and cache key available
        if use_cache and cache_key:
            self._save_to_cache(cache_key, results)
        
        return results
    
    def _print_processing_summary(self, results):
        """Print a single-line processing summary."""
        metadata = results['metadata']
        source_file = metadata.get('source_file', 'unknown')
        
        # Point statistics
        original_count = metadata['original_shape'][0]
        final_count = len(results['processed_points'])
        removed_count = metadata.get('points_removed', 0)
        
        # Format: 1k, 10k, etc.
        def format_count(n):
            if n >= 1000:
                return f"{n//1000}k"
            return str(n)
        
        summary_parts = [f"{source_file}: {format_count(final_count)} points"]
        
        if removed_count > 0:
            summary_parts.append(f"(-{removed_count} outliers)")
        
        # Border weight statistics
        if 'border_weight_stats' in metadata:
            stats = metadata['border_weight_stats']
            summary_parts.append(f"{stats['high']} high {stats['medium']} medium {stats['low']} low borderweights")
        
        # Group statistics with per-group point counts
        if 'num_groups' in metadata and 'group_info' in metadata:
            group_info = metadata['group_info']
            num_groups = metadata['num_groups']
            # Get point counts per group (use 'cleaned_size' key)
            group_sizes = [info['cleaned_size'] for info in group_info]
            group_sizes_str = ', '.join(map(str, group_sizes))
            summary_parts.append(f"{num_groups} groupids ({group_sizes_str}) points")
        
        print(', '.join(summary_parts))
    
    def _apply_surface_aware_processing(self, results, verbose):
        """Apply surface-aware processing pipeline with sequential steps."""
        from .surface_grouping import simple_spatial_grouping, process_all_surface_groups
        
        point_cloud = results['processed_points']
        wireframe_vertices = results['processed_wireframe']
        original_count = len(point_cloud)
        
        # Step 1: Spatial Grouping (on raw unnormalized points)
        grouping_params = self.config['grouping_params']['coarse_params']
        surface_groups = simple_spatial_grouping(
            point_cloud[:, 0:3],
            **grouping_params
        )
        
        if not surface_groups:
            if verbose:
                print("Warning: No surface groups detected, falling back to standard normalization")
            return self._apply_standard_normalization(results, verbose)
        
        results['metadata']['num_groups'] = len(surface_groups)
        
        # Step 2: Per-Group Outlier Removal
        processing_params = {
            'apply_outlier_removal': self.config['use_outlier_removal'],
            'compute_normals': False,
            'use_neighbor_cache': True,
            'config': self.config['grouping_params']
        }
        
        if self.config['use_outlier_removal']:
            processing_params['outlier_removal_params'] = self.config['outlier_params']
        
        processing_results = process_all_surface_groups(
            surface_groups,
            **processing_params
        )
        
        cleaned_points = processing_results['cleaned_points']
        group_ids = processing_results['group_ids']
        removed_count = original_count - len(cleaned_points)
        
        results['metadata']['points_removed'] = removed_count
        results['metadata']['group_info'] = processing_results['group_info']
        
        # Step 3: Normalization (after cleaning)
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
        
        # Reconstruct point cloud with X Y Z GroupID
        reconstructed_pc = np.zeros((len(cleaned_points), 4))
        reconstructed_pc[:, 0:3] = normalized_coords
        reconstructed_pc[:, 3] = group_ids
        
        results['processed_points'] = reconstructed_pc
        results['metadata']['centroid'] = centroid
        results['metadata']['max_distance'] = max_distance
        results['metadata']['has_group_ids'] = True
        results['metadata']['processing_steps'].append('surface_aware_processing')
        
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
        
        # Extract coordinates only
        coordinates = point_cloud[:, 0:3]
        
        # Compute border weights
        weight_params = self.config['weight_params']
        border_weights = compute_border_weights(
            coordinates,
            k_neighbors=weight_params['k_neighbors'],
            normalize_weights=weight_params['normalize_weights'],
            multi_scale=weight_params['multi_scale']
        )
        
        # Categorize points by border weight
        high_weight = np.sum(border_weights > 0.7)
        medium_weight = np.sum((border_weights >= 0.3) & (border_weights <= 0.7))
        low_weight = np.sum(border_weights < 0.3)
        
        results['metadata']['border_weight_stats'] = {
            'high': high_weight,
            'medium': medium_weight,
            'low': low_weight,
            'mean': float(np.mean(border_weights)),
            'std': float(np.std(border_weights))
        }
        
        # Add border weights to the point cloud
        results['processed_points'] = np.column_stack([point_cloud, border_weights])
        results['metadata']['has_border_weights'] = True
        results['metadata']['border_weight_column'] = point_cloud.shape[1]
        results['metadata']['processing_steps'].append('border_weights')
        
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