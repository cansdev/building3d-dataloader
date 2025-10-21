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


def calculate_input_dim(dataset_config):
    """
    Dynamically calculate input dimensions based on enabled features.
    This function computes the expected number of channels in the point cloud input.
    
    Args:
        dataset_config: Dataset configuration object
        
    Returns:
        input_dim: Total number of input channels
    """
    input_dim = 3  # XYZ coordinates (always present)
    
    # Add color channels
    if dataset_config.use_color:
        input_dim += 4  # RGBA
    
    # Add intensity channel
    if dataset_config.use_intensity:
        input_dim += 1  # Intensity
    
    # Add geometric features if enabled
    if getattr(dataset_config, 'use_group_ids', False):
        input_dim += 1  # Normalized group IDs
    
    if getattr(dataset_config, 'use_border_weights', False):
        input_dim += 1  # Border weights
    
    return input_dim


class Building3DPreprocessor:
    """
    Handles one-time preprocessing and caching for Building3D dataset.
    
    Pipeline:
        1. Normalize (centroid + scale)
        2. Spatial Grouping (DBSCAN-based surface segmentation) - ALWAYS computed
        3. Outlier Detection (per-group statistical + radius filtering) - ALWAYS computed, saved as boolean mask
        4. Border Weight Assignment (edge + vertex + border weights) - ALWAYS computed
        5. Cache ALL data to .npz file
        
    Note: ALL features are computed and cached regardless of config.
    Config flags only control what gets loaded/used during training.
    """
    def __init__(self, dataset_config):
        self.config = dataset_config
        self.cache_dir = dataset_config.preprocessor.cache_dir
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def preprocess_and_cache(self, pc_file, split_set):
        """
        Preprocess a single point cloud and cache ALL results regardless of config.
        
        ALWAYS computes and caches:
        - Raw coordinates (XYZ)
        - Raw RGBA colors
        - Raw intensity
        - Normalized coordinates (XYZ)
        - Normalized RGBA colors
        - Normalized intensity
        - Group IDs (surface segmentation)
        - Border weights (edge detection)
        - Outlier mask (boolean array indicating outliers)
        
        Args:
            pc_file: Path to .xyz file
            split_set: 'train' or 'test'
            
        Returns:
            cache_path: Path to cached .npz file
        """
        # Generate cache path
        cache_path = self._get_cache_path(pc_file, split_set)
        
        # If already cached, skip
        if os.path.exists(cache_path):
            return cache_path
            
        # Load raw point cloud - ALWAYS load all 8 channels (XYZ + RGBA + Intensity)
        pc = np.loadtxt(pc_file, dtype=np.float64)
        
        # Separate raw features
        raw_xyz = pc[:, 0:3].copy()
        raw_rgba = pc[:, 3:7].copy() if pc.shape[1] >= 7 else np.zeros((len(pc), 4), dtype=np.float64)
        raw_intensity = pc[:, 7:8].copy() if pc.shape[1] >= 8 else np.zeros((len(pc), 1), dtype=np.float64)
        
        # Step 1: Normalize - create normalized versions
        centroid, max_distance, normalized_xyz, normalized_rgba, normalized_intensity = self._normalize(
            raw_xyz, raw_rgba, raw_intensity
        )
        
        # Step 2: Spatial Grouping - ALWAYS compute
        group_ids = self._spatial_grouping(normalized_xyz)
        
        # Step 3: Outlier Detection - ALWAYS compute, save as boolean mask
        outlier_mask = self._detect_outliers(normalized_xyz, group_ids)
        
        # Step 4: Border Weight Assignment - ALWAYS compute
        edge_weights, edgecross_weights, border_weights = self._compute_weights(normalized_xyz)
        
        # Step 5: Save ALL data to cache
        self._save_cache(
            cache_path,
            # Raw data
            raw_xyz=raw_xyz.astype(np.float32),
            raw_rgba=raw_rgba.astype(np.float32),
            raw_intensity=raw_intensity.astype(np.float32),
            # Normalized data
            normalized_xyz=normalized_xyz.astype(np.float32),
            normalized_rgba=normalized_rgba.astype(np.float32),
            normalized_intensity=normalized_intensity.astype(np.float32),
            # Geometric features
            group_ids=group_ids.astype(np.int32),
            edge_weights=edge_weights.astype(np.float32),
            edgecross_weights=edgecross_weights.astype(np.float32),
            border_weights=border_weights.astype(np.float32),
            # Outlier detection
            outlier_mask=outlier_mask.astype(bool),
            # Normalization parameters
            centroid=centroid.astype(np.float32),
            max_distance=np.float32(max_distance)
        )
        
        return cache_path
    
    def _normalize(self, xyz, rgba, intensity):
        """
        Step 1: Normalize point cloud components separately.
        
        Returns:
            centroid: XYZ centroid
            max_distance: Max distance from centroid
            normalized_xyz: XYZ normalized to unit sphere
            normalized_rgba: RGBA normalized to [0, 1]
            normalized_intensity: Intensity normalized to [0, 1]
        """
        # Normalize XYZ coordinates
        centroid = np.mean(xyz, axis=0)
        xyz_centered = xyz - centroid
        max_distance = np.max(np.linalg.norm(xyz_centered, axis=1))
        
        if max_distance < 1e-6:
            max_distance = 1.0
            
        normalized_xyz = xyz_centered / max_distance
        
        # Normalize RGBA to [0, 1] range
        normalized_rgba = rgba / 256.0
        
        # Normalize intensity to [0, 1] range (assuming 16-bit: 0-65535)
        # Handle edge case where intensity might be all zeros
        intensity_max = np.max(intensity)
        if intensity_max > 0:
            normalized_intensity = intensity / 65535.0  # 16-bit normalization
        else:
            normalized_intensity = intensity.copy()
        
        return centroid, max_distance, normalized_xyz, normalized_rgba, normalized_intensity
    
    def _spatial_grouping(self, points):
        """
        Step 2: Spatial grouping using surface_grouping.py
        ALWAYS computed regardless of config.
        """
        params = self.config.preprocessor.grouping_params.coarse_params
        
        try:
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
        except Exception as e:
            print(f"Warning: Spatial grouping failed: {e}, using default group 0")
            group_ids = np.zeros(len(points), dtype=np.int32)
            
        return group_ids
    
    def _detect_outliers(self, points, group_ids):
        """
        Step 3: Detect outliers per group using outlier_removal.py
        ALWAYS computed regardless of config.
        
        Returns:
            outlier_mask: Boolean array where True = outlier, False = inlier
        """
        params = self.config.preprocessor.outlier_params
        
        # Initialize as all inliers
        outlier_mask = np.zeros(len(points), dtype=bool)
        
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
                
                # Mark outliers in global mask
                group_indices = np.where(group_mask)[0]
                outlier_mask[group_indices[~inlier_mask]] = True
            except Exception as e:
                print(f"Warning: Outlier detection failed for group {group_id}: {e}")
                continue
        
        return outlier_mask
    
    def _compute_weights(self, points):
        """
        Step 4: Compute border weights using weight_assignment.py
        ALWAYS computed regardless of config.
        """
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
            print(f"Warning: Weight computation failed: {e}, using zeros")
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
        
        # Config flags to control what features to USE (all are always COMPUTED)
        self.use_group_ids = getattr(dataset_config, 'use_group_ids', False)
        self.use_border_weights = getattr(dataset_config, 'use_border_weights', False)
        self.use_outlier_removal = dataset_config.preprocessor.use_outlier_removal
        
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
        ALL features are computed regardless of config.
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
                logger.info(f"Preprocessing {len(files_to_process)} files (computing ALL features)...")
            
            # Preprocess with progress bar
            for pc_file in tqdm(files_to_process, desc="Preprocessing", disable=not logger):
                try:
                    self.preprocessor.preprocess_and_cache(pc_file, self.split_set)
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
        
        # Load from cache - ALL data is available
        point_cloud, centroid, max_distance, group_ids, border_weights, outlier_mask = self._load_from_cache(pc_file)
        
        # Apply outlier filtering if enabled in config
        if self.use_outlier_removal:
            inlier_mask = ~outlier_mask
            point_cloud = point_cloud[inlier_mask]
            group_ids = group_ids[inlier_mask]
            border_weights = border_weights[inlier_mask]
        
        # ------------------------------- Load Wireframe ------------------------------
        wf_vertices, wf_edges = load_wireframe(wireframe_file)
        
        # Only normalize wireframe if normalize flag is True
        if self.normalize:
            wf_vertices = (wf_vertices - centroid) / max_distance
        
        # ------------------------------- Random Sampling (Per Epoch) ------------------------------
        if self.num_points:
            current_points = len(point_cloud)
            
            if current_points > self.num_points:
                # For test data, use deterministic sampling for reproducibility
                if self.split_set == "test":
                    # Use scan_idx as seed for deterministic sampling
                    scan_idx = int(os.path.splitext(os.path.basename(pc_file))[0])
                    np.random.seed(scan_idx)
                # Downsample: random sampling without replacement
                indices = np.random.choice(current_points, self.num_points, replace=False)
            elif current_points < self.num_points:
                # For test data, use deterministic sampling for reproducibility
                if self.split_set == "test":
                    # Use scan_idx as seed for deterministic sampling
                    scan_idx = int(os.path.splitext(os.path.basename(pc_file))[0])
                    np.random.seed(scan_idx)
                # Upsample: random sampling with replacement to reach target count
                indices = np.random.choice(current_points, self.num_points, replace=True)
            else:
                # Exact match: use all points
                indices = np.arange(current_points)
            
            # Apply sampling to all arrays
            point_cloud = point_cloud[indices]
            group_ids = group_ids[indices]
            border_weights = border_weights[indices]

        # ------------------------------- Augmentation (Per Epoch) ------------------------------
        # Disable augmentation for test data to ensure reproducibility
        if self.augment and self.split_set == "train":
            point_cloud, wf_vertices = self._apply_augmentation(point_cloud, wf_vertices)
        
        # ------------------------------- Concatenate Geometric Features ------------------------------
        # Optionally concatenate group_ids and border_weights as additional input channels
        if self.use_group_ids:
            # Normalize group_ids using FIXED max across entire dataset
            group_ids_normalized = np.clip(group_ids.astype(np.float32) / self.MAX_GROUPS, 0.0, 1.0)
            point_cloud = np.concatenate([point_cloud, group_ids_normalized[:, np.newaxis]], axis=1)
        
        if self.use_border_weights:
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
        
        # Geometric features (always available in cache, returned for debugging/analysis)
        ret_dict['group_ids'] = group_ids.astype(np.int32)
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
        """
        Load preprocessed data from cache.
        
        Assembles point cloud based on config flags:
        - normalize: use normalized vs raw coordinates
        - use_color: include RGBA channels
        - use_intensity: include intensity channel
        
        Returns:
            point_cloud: Assembled point cloud with selected features [N, C]
            centroid: Normalization centroid [3]
            max_distance: Normalization scale factor
            group_ids: Surface group IDs [N]
            border_weights: Border detection weights [N]
            outlier_mask: Boolean mask of outliers [N]
        """
        cache_path = self.preprocessor._get_cache_path(pc_file, self.split_set)
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cache file not found: {cache_path}")
        
        # Load cached data
        cached = np.load(cache_path)
        
        # Select coordinates based on normalize flag
        if self.normalize:
            xyz = cached['normalized_xyz']
            rgba = cached['normalized_rgba']
            intensity = cached['normalized_intensity']
        else:
            xyz = cached['raw_xyz']
            rgba = cached['raw_rgba']
            intensity = cached['raw_intensity']
        
        # Assemble point cloud based on config
        components = [xyz]
        
        if self.use_color:
            components.append(rgba)
        
        if self.use_intensity:
            components.append(intensity)
        
        point_cloud = np.concatenate(components, axis=1)
        
        # Load other cached data
        centroid = cached['centroid']
        max_distance = cached['max_distance']
        group_ids = cached['group_ids']
        border_weights = cached['border_weights']
        outlier_mask = cached['outlier_mask']
        
        return point_cloud, centroid, max_distance, group_ids, border_weights, outlier_mask
    
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
                elif key in ['group_ids', 'border_weights']:
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
