#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py.py
# @IDE     : PyCharm

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

# Import preprocessing utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.surface_grouping import simple_spatial_grouping
from utils.outlier_removal import remove_statistical_outliers, remove_radius_outliers
from utils.weight_assignment import compute_border_weights, calculate_local_features
from utils.weight_assignment import calculate_edge_weights, calculate_edgecross_weights


def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            # Skip empty lines and comments
            if not line or not line[0] or line[0].startswith('#'):
                continue
            if line[0] == 'v':
                vertices.append(line[1:])
            elif line[0] == 'l':
                obj_data = np.array(line[1:], dtype=np.int32).reshape(2) - 1
                edges.add(tuple(sorted(obj_data)))
    vertices = np.array(vertices, dtype=np.float64)
    edges = np.array(list(edges))
    return vertices, edges


def save_wireframe(vertices, edges, wireframe_file):
    r"""
    :param wireframe_file: wireframe file name
    :param vertices: N * 3, vertex coordinates
    :param edges: M * 2,
    :return:
    """
    with open(wireframe_file, 'w') as f:
        for vertex in vertices:
            line = ' '.join(map(str, vertex))
            f.write('v ' + line + '\n')
        for edge in edges:
            edge = ' '.join(map(str, edge + 1))
            f.write('l ' + edge + '\n')


def random_sampling(pc, num_points, replace=None, return_choices=False):
    r"""
    :param pc: N * 3
    :param num_points: Int
    :param replace:
    :param return_choices:
    :return:
    """
    if replace is None:
        replace = pc.shape[0] < num_points
    choices = np.random.choice(pc.shape[0], num_points, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Building3DPreprocessor:
    """
    Handles one-time preprocessing and caching for Building3D dataset.
    
    Pipeline:
        1. Normalize (centroid + scale)
        2. Spatial Grouping (DBSCAN-based surface segmentation)
        3. Outlier Removal (per-group statistical + radius filtering)
        4. Border Weight Assignment (edge + vertex + border weights)
        5. Cache to .npz file
    """
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.cache_dir = dataset_config.preprocessor.cache_dir
        self.use_outlier_removal = dataset_config.preprocessor.use_outlier_removal
        
        # Use unified flags from top-level config
        # If feature is enabled for model input, we must compute it during preprocessing
        self.use_surface_grouping = getattr(dataset_config, 'use_group_ids', False)
        self.compute_border_weights = getattr(dataset_config, 'use_border_weights', False)
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def preprocess_and_cache(self, pc_file, split_set, use_color=True, use_intensity=True):
        """
        Preprocess a single point cloud and cache the result.
        
        Args:
            pc_file: Path to .xyz file
            split_set: 'train' or 'test'
            use_color: Whether to include RGBA channels
            use_intensity: Whether to include intensity channel
            
        Returns:
            cache_path: Path to cached .npz file
        """
        # Generate cache path
        cache_path = self._get_cache_path(pc_file, split_set)
        
        # If already cached, skip
        if os.path.exists(cache_path):
            return cache_path
            
        # Load raw point cloud
        pc = np.loadtxt(pc_file, dtype=np.float64)
        
        # Extract features based on config
        if not use_color:
            point_cloud = pc[:, 0:3]
        elif use_color and not use_intensity:
            point_cloud = pc[:, 0:7]
        elif not use_color and use_intensity:
            point_cloud = np.concatenate((pc[:, 0:3], pc[:, 7:8]), axis=1)
        else:
            point_cloud = pc
            
        # Step 1: Normalize
        centroid, max_distance, normalized_pc = self._normalize(point_cloud)
        
        # Step 2: Spatial Grouping (if enabled)
        if self.use_surface_grouping:
            group_ids = self._spatial_grouping(normalized_pc[:, :3])
        else:
            group_ids = np.zeros(len(normalized_pc), dtype=np.int32)
            
        # Step 3: Outlier Removal (per group if enabled)
        if self.use_outlier_removal:
            clean_mask = self._remove_outliers(normalized_pc, group_ids)
            normalized_pc = normalized_pc[clean_mask]
            group_ids = group_ids[clean_mask]
            point_cloud = point_cloud[clean_mask]  # FIX: Apply mask to raw point cloud too
            
        # Step 4: Border Weight Assignment
        if self.compute_border_weights:
            edge_weights, edgecross_weights, border_weights = self._compute_weights(
                normalized_pc[:, :3]
            )
        else:
            N = len(normalized_pc)
            edge_weights = np.zeros(N, dtype=np.float32)
            edgecross_weights = np.zeros(N, dtype=np.float32)
            border_weights = np.zeros(N, dtype=np.float32)
            
        # Step 5: Save to cache with BOTH raw and normalized coordinates
        self._save_cache(
            cache_path,
            raw_pc=point_cloud.astype(np.float32),
            normalized_pc=normalized_pc.astype(np.float32),
            group_ids=group_ids.astype(np.int32),
            edge_weights=edge_weights.astype(np.float32),
            edgecross_weights=edgecross_weights.astype(np.float32),
            border_weights=border_weights.astype(np.float32),
            centroid=centroid.astype(np.float32),
            max_distance=np.float32(max_distance)
        )
        
        return cache_path
    
    def _normalize(self, pc):
        """Step 1: Normalize point cloud (centroid + scale)"""
        xyz = pc[:, :3]
        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid
        max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
        
        if max_distance < 1e-6:
            max_distance = 1.0
            
        xyz_normalized = xyz_centered / max_distance
        
        # Rebuild point cloud with normalized coords
        normalized_pc = pc.copy()
        normalized_pc[:, :3] = xyz_normalized
        
        # Normalize colors if present (RGBA: columns 3-6)
        if normalized_pc.shape[1] >= 7:
            normalized_pc[:, 3:7] = normalized_pc[:, 3:7] / 256.0
            
        return centroid, max_distance, normalized_pc
    
    def _spatial_grouping(self, points):
        """Step 2: Spatial grouping using surface_grouping.py"""
        params = self.config.preprocessor.grouping_params.coarse_params
        
        surface_groups = simple_spatial_grouping(
            points,
            eps=params.eps,
            min_samples=params.min_samples,
            auto_scale_eps=params.auto_scale_eps
        )
        
        # Create group_id array
        group_ids = np.zeros(len(points), dtype=np.int32)
        for i, group in enumerate(surface_groups):
            group_ids[group.indices] = i
            
        return group_ids
    
    def _remove_outliers(self, pc, group_ids):
        """Step 3: Outlier removal per group using outlier_removal.py"""
        params = self.config.preprocessor.outlier_params
        points = pc[:, :3]
        
        # Apply outlier removal per group
        keep_mask = np.ones(len(points), dtype=bool)
        
        for group_id in np.unique(group_ids):
            group_mask = group_ids == group_id
            group_points = points[group_mask]
            
            if len(group_points) < 10:
                continue
                
            # Statistical outlier removal
            try:
                _, inlier_mask = remove_statistical_outliers(
                    group_points,
                    k_neighbors=params.stat_k,
                    std_ratio=params.stat_std_ratio
                )
                
                # Update global mask
                group_indices = np.where(group_mask)[0]
                keep_mask[group_indices[~inlier_mask]] = False
            except Exception as e:
                print(f"Warning: Outlier removal failed for group {group_id}: {e}")
                continue
        
        return keep_mask
    
    def _compute_weights(self, points):
        """Step 4: Compute border weights using weight_assignment.py"""
        params = self.config.preprocessor.weight_params
        
        try:
            # Compute eigenvalues
            eigenvalues = calculate_local_features(points, params.k_neighbors)
            
            # Compute individual weights
            edge_weights = calculate_edge_weights(eigenvalues)
            edgecross_weights = calculate_edgecross_weights(eigenvalues)
            
            # Compute unified border weights
            border_weights = compute_border_weights(
                points,
                k_neighbors=params.k_neighbors,
                normalize_weights=params.normalize_weights,
                multi_scale=params.multi_scale
            )
        except Exception as e:
            print(f"Warning: Weight computation failed: {e}")
            N = len(points)
            edge_weights = np.zeros(N, dtype=np.float32)
            edgecross_weights = np.zeros(N, dtype=np.float32)
            border_weights = np.zeros(N, dtype=np.float32)
        
        return edge_weights, edgecross_weights, border_weights
    
    def _save_cache(self, cache_path, **kwargs):
        """Save preprocessed data to .npz file"""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(cache_path, **kwargs)
        
    def _get_cache_path(self, pc_file, split_set):
        """
        Generate cache file path from original .xyz file.
        
        Example:
            Input: datasets/demo_dataset/train/xyz/1.xyz
            Output: datasets/preprocessed/train/1.npz
        """
        filename = os.path.splitext(os.path.basename(pc_file))[0]
        return os.path.join(self.cache_dir, split_set, f"{filename}.npz")


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        self.normalize = dataset_config.normalize
        self.augment = dataset_config.augment
        
        # Option to include geometric features as input channels
        self.use_group_ids = getattr(dataset_config, 'use_group_ids', False)
        self.use_border_weights = getattr(dataset_config, 'use_border_weights', False)
        
        # Fixed normalization constant for group IDs (based on dataset analysis)
        # Max observed group_id is 4, so we use 5 to handle 0-4 range consistently
        self.MAX_GROUPS = 5

        assert split_set in ["train", "test"]
        self.split_set = split_set

        self.pc_files, self.wireframe_files = self.load_files()

        if logger:
            logger.info("Total Sample: %d" % len(self.pc_files))
            
        self.preprocessor = Building3DPreprocessor(dataset_config)
        self._ensure_preprocessed(logger)
    
    def _ensure_preprocessed(self, logger):
        """
        Preprocess all files in the dataset if not already cached.
        This runs once when dataset is initialized.
        """
        if logger:
            logger.info("Checking preprocessed cache...")
        
        files_to_process = []
        for pc_file in self.pc_files:
            cache_path = self.preprocessor._get_cache_path(pc_file, self.split_set)
            if not os.path.exists(cache_path):
                files_to_process.append(pc_file)
        
        if files_to_process:
            if logger:
                logger.info(f"Preprocessing {len(files_to_process)} files...")
            
            # Preprocess with progress bar
            for pc_file in tqdm(files_to_process, desc="Preprocessing", disable=not logger):
                try:
                    self.preprocessor.preprocess_and_cache(
                        pc_file, 
                        self.split_set,
                        use_color=self.use_color,
                        use_intensity=self.use_intensity
                    )
                except Exception as e:
                    if logger:
                        logger.error(f"Failed to preprocess {os.path.basename(pc_file)}: {e}")
                    raise
            
            if logger:
                logger.info("All files preprocessed and cached!")
        else:
            if logger:
                logger.info("All files already cached - skipping preprocessing")

    def __len__(self):
        return len(self.pc_files)

    def __getitem__(self, index):
        pc_file = self.pc_files[index]
        wireframe_file = self.wireframe_files[index]
        
        point_cloud, centroid, max_distance, group_ids, edge_weights, edgecross_weights, border_weights = self._load_from_cache(pc_file)
        
        # ------------------------------- Load Wireframe ------------------------------
        wf_vertices, wf_edges = load_wireframe(wireframe_file)
        
        # NEW: Only normalize wireframe if normalize flag is True
        if self.normalize:
            wf_vertices = (wf_vertices - centroid) / max_distance
        
        # ------------------------------- Random Sampling (Per Epoch) ------------------------------
        if self.num_points:
            current_points = len(point_cloud)
            
            if current_points > self.num_points:
                # Downsample: random sampling without replacement
                indices = np.random.choice(current_points, self.num_points, replace=False)
            elif current_points < self.num_points:
                # Upsample: random sampling with replacement to reach target count
                indices = np.random.choice(current_points, self.num_points, replace=True)
            else:
                # Exact match: use all points
                indices = np.arange(current_points)
            
            # Apply sampling to point cloud
            point_cloud = point_cloud[indices]
            
            # Sample geometric features if available
            if group_ids is not None:
                group_ids = group_ids[indices]
            if edge_weights is not None:
                edge_weights = edge_weights[indices]
            if edgecross_weights is not None:
                edgecross_weights = edgecross_weights[indices]
            if border_weights is not None:
                border_weights = border_weights[indices]

        # ------------------------------- Augmentation (Per Epoch) ------------------------------
        if self.augment:
            point_cloud, wf_vertices = self._apply_augmentation(point_cloud, wf_vertices)
        
        # ------------------------------- Concatenate Geometric Features ------------------------------
        # Optionally concatenate group_ids and border_weights as additional input channels
        if self.use_group_ids and group_ids is not None:
            # Normalize group_ids using FIXED max across entire dataset
            # This ensures consistent scaling: same group_id value means same thing across all samples
            group_ids_normalized = np.clip(group_ids.astype(np.float32) / self.MAX_GROUPS, 0.0, 1.0)
            point_cloud = np.concatenate([point_cloud, group_ids_normalized[:, np.newaxis]], axis=1)
        
        if self.use_border_weights and border_weights is not None:
            # Border weights are already in [0, 1] range
            point_cloud = np.concatenate([point_cloud, border_weights[:, np.newaxis]], axis=1)

        # -------------------------------Edge Vertices ------------------------
        wf_edges_vertices = np.stack((wf_vertices[wf_edges[:, 0]], wf_vertices[wf_edges[:, 1]]), axis=1)
        wf_edges_vertices = wf_edges_vertices[
            np.arange(wf_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(wf_edges_vertices[:, :, -1]),
                                                                        axis=1)]
        wf_centers = (wf_edges_vertices[..., 0, :] + wf_edges_vertices[..., 1, :]) / 2
        wf_edge_number = wf_edges.shape[0]

        # ------------------------------- Return Dict ------------------------------
        ret_dict = {}
        
        # Point cloud features
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        
        # NEW: Geometric features (if available)
        if group_ids is not None:
            ret_dict['group_ids'] = group_ids.astype(np.int32)
        if edge_weights is not None:
            ret_dict['edge_weights'] = edge_weights.astype(np.float32)
        if edgecross_weights is not None:
            ret_dict['edgecross_weights'] = edgecross_weights.astype(np.float32)
        if border_weights is not None:
            ret_dict['border_weights'] = border_weights.astype(np.float32)
        
        # Wireframe data
        ret_dict['wf_vertices'] = wf_vertices.astype(np.float32)
        ret_dict['wf_edges'] = wf_edges.astype(np.int64)
        ret_dict['wf_centers'] = wf_centers.astype(np.float32)
        ret_dict['wf_edge_number'] = wf_edge_number
        ret_dict['wf_edges_vertices'] = wf_edges_vertices.reshape((-1, 6)).astype(np.float32)
        
        # Normalization parameters
        if self.normalize:
            ret_dict['centroid'] = centroid.astype(np.float32)
            ret_dict['max_distance'] = np.float32(max_distance)
            
        ret_dict['scan_idx'] = np.array(os.path.splitext(os.path.basename(pc_file))[0]).astype(np.int64)
        return ret_dict
    
    def _load_from_cache(self, pc_file):
        """Load preprocessed data from cache"""
        cache_path = self.preprocessor._get_cache_path(pc_file, self.split_set)
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        # Load cached data
        cached = np.load(cache_path)
        
        # Choose between raw and normalized coordinates based on normalize flag
        if self.normalize:
            point_cloud = cached['normalized_pc']
        else:
            if 'raw_pc' in cached:
                point_cloud = cached['raw_pc']
                # Still need to normalize colors if present
                if point_cloud.shape[1] >= 7:
                    point_cloud = point_cloud.copy()  # Don't modify cached data
                    point_cloud[:, 3:7] = point_cloud[:, 3:7] / 256.0
            else:
                print("Warning: raw_pc not found in cache, using normalized_pc")
                point_cloud = cached['normalized_pc']
        
        centroid = cached['centroid']
        max_distance = cached['max_distance']
        group_ids = cached['group_ids']
        edge_weights = cached['edge_weights']
        edgecross_weights = cached['edgecross_weights']
        border_weights = cached['border_weights']
        
        return point_cloud, centroid, max_distance, group_ids, edge_weights, edgecross_weights, border_weights
    
    def _apply_augmentation(self, point_cloud, wf_vertices):
        """Apply data augmentation (flip + rotation)"""
        if np.random.random() > 0.5:
            # Flipping along the YZ plane
            point_cloud[:, 0] = -1 * point_cloud[:, 0]
            wf_vertices[:, 0] = -1 * wf_vertices[:, 0]

        if np.random.random() > 0.5:
            # Flipping along the XZ plane
            point_cloud[:, 1] = -1 * point_cloud[:, 1]
            wf_vertices[:, 1] = -1 * wf_vertices[:, 1]

        # Rotation along up-axis/Z-axis
        rot_angle = (np.random.random() * 2 * np.pi) - np.pi  # -180 ~ +180 degree
        rot_mat = rotz(rot_angle)
        point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
        wf_vertices[:, 0:3] = np.dot(wf_vertices[:, 0:3], np.transpose(rot_mat))
        
        return point_cloud, wf_vertices

    @staticmethod
    def collate_batch(batch):
        input_dict = defaultdict(list)
        for item in batch:
            for key, val in item.items():
                input_dict[key].append(val)

        ret_dict = {}
        for key, val in input_dict.items():
            try:
                if key in ['wf_vertices', 'wf_edges', 'wf_centers', 'wf_edges_vertices']:
                    # For Hungarian matching, we don't need padding - just return as list
                    # The training loop will handle variable lengths
                    ret_dict[key] = [torch.from_numpy(v.astype(np.float32)) for v in val]
                elif key in ['group_ids', 'edge_weights', 'edgecross_weights', 'border_weights']:
                    # Handle geometric features (per-point features)
                    # These are separate from point_clouds and should be stacked directly
                    ret_dict[key] = torch.tensor(np.array(input_dict[key]))
                elif key == 'point_clouds':
                    # Handle point_clouds specially - stack along batch dimension
                    # All samples should have same shape [N, C] after preprocessing
                    ret_dict[key] = torch.from_numpy(np.stack(val, axis=0).astype(np.float32))
                else:
                    ret_dict[key] = torch.tensor(np.array(input_dict[key]))
            except Exception as e:
                print(f'Error in collate_batch: key={key}, error={e}')
                print(f'  Value shapes: {[v.shape if hasattr(v, "shape") else type(v) for v in val[:3]]}')
                raise TypeError

        return ret_dict

    def load_files(self):
        data_dir = os.path.join(self.roof_dir, self.split_set)
        pc_files = [pc_file for pc_file in glob.glob(os.path.join(data_dir, 'xyz', '*.xyz'))]
        wireframe_files = [wireframe_file.replace(os.path.sep + "xyz", os.path.sep + "wireframe").replace(".xyz", ".obj") for wireframe_file in
                           pc_files]
        return pc_files, wireframe_files

    def print_self_values(self):
        attributes = vars(self)
        for attribute, value in attributes.items():
            print(attribute, "=", value)
