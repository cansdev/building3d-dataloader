#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 3:06 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : building3d.py
# @IDE     : PyCharm

import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from utils.preprocess import create_default_preprocessor

def load_wireframe(wireframe_file):
    vertices = []
    edges = set()
    with open(wireframe_file) as f:
        for lines in f.readlines():
            line = lines.strip().split(' ')
            if line[0] == 'v':
                vertices.append(line[1:])
            else:
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


def random_sampling(pc, num_points, replace=None, return_choices=False, seed=None):
    r"""
    Deterministic random sampling with optional seed for reproducibility.
    :param pc: N * 3
    :param num_points: Int
    :param replace:
    :param return_choices:
    :param seed: Random seed for deterministic sampling
    :return:
    """
    if replace is None:
        replace = pc.shape[0] < num_points
    
    # Set seed for deterministic sampling if provided
    if seed is not None:
        np.random.seed(seed)
    
    choices = np.random.choice(pc.shape[0], num_points, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def farthest_point_sampling(pc, num_points, seed=None):
    r"""
    Farthest Point Sampling for better point cloud coverage.
    Selects points that are maximally distant from each other.
    
    :param pc: N * D (point cloud with N points and D features, uses first 3 for distance)
    :param num_points: Int (number of points to sample)
    :param seed: Random seed for deterministic first point selection
    :return: Sampled point cloud of shape (num_points, D)
    """
    N, D = pc.shape
    
    # If requesting more points than available, duplicate points to reach desired count
    if num_points >= N:
        # Use random sampling with replacement to reach num_points
        if seed is not None:
            np.random.seed(seed)
        indices = np.random.choice(N, num_points, replace=True)
        return pc[indices]
    
    # Initialize
    xyz = pc[:, :3]  # Use only XYZ for distance computation
    centroids = np.zeros(num_points, dtype=np.int32)
    distances = np.ones(N) * 1e10
    
    # Select first point randomly (with seed for reproducibility)
    if seed is not None:
        np.random.seed(seed)
    farthest = np.random.randint(0, N)
    
    # Iteratively select farthest points
    for i in range(num_points):
        centroids[i] = farthest
        centroid_xyz = xyz[farthest, :]
        
        # Compute distances from current centroid to all points
        dist = np.sum((xyz - centroid_xyz) ** 2, axis=1)
        
        # Update minimum distances
        mask = dist < distances
        distances[mask] = dist[mask]
        
        # Select the farthest point
        farthest = np.argmax(distances)
    
    # Return sampled points with all features
    return pc[centroids]


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        # Read parameters from preprocessor section (moved there to avoid duplication)
        self.normalize = dataset_config.preprocessor.normalize
        self.augment = dataset_config.augment
        self.use_surface_grouping = dataset_config.preprocessor.use_surface_grouping
        self.use_outlier_removal = dataset_config.preprocessor.use_outlier_removal
        
        # Epoch counter for deterministic-but-varying random sampling
        self.epoch = 0
        
        # Flag to show verbose output only on first epoch
        self.show_preprocessing_details = True
        
        # Initialize preprocessor using YAML configuration
        self.preprocessor = create_default_preprocessor()
        
        # Override specific settings from dataset config (now in preprocessor section)
        if hasattr(dataset_config.preprocessor, 'use_outlier_removal'):
            self.preprocessor.config['use_outlier_removal'] = dataset_config.preprocessor.use_outlier_removal
        if hasattr(dataset_config.preprocessor, 'normalize'):
            self.preprocessor.config['normalize'] = dataset_config.preprocessor.normalize
        if hasattr(dataset_config.preprocessor, 'use_surface_grouping'):
            self.preprocessor.config['use_surface_grouping'] = dataset_config.preprocessor.use_surface_grouping

        assert split_set in ["train", "test"]
        self.split_set = split_set

        self.pc_files, self.wireframe_files = self.load_files()

        if logger:
            logger.info("Total Sample: %d" % len(self.pc_files))
            if self.use_surface_grouping:
                logger.info("Using surface-aware processing pipeline")
            logger.info("Outlier removal: %s" % ("Enabled" if self.use_outlier_removal else "Disabled"))

    def __len__(self):
        return len(self.pc_files)
    
    def set_epoch(self, epoch):
        """Set the epoch number for deterministic-but-varying random sampling."""
        self.epoch = epoch
        # Only show preprocessing details on first epoch (epoch 0)
        self.show_preprocessing_details = (epoch == 0)

    def __getitem__(self, index):
        # ------------------------------- Point Clouds ------------------------------
        # load point clouds
        pc_file = self.pc_files[index]
        pc = np.loadtxt(pc_file, dtype=np.float64)

        # point clouds processing
        if not self.use_color:
            point_cloud = pc[:, 0:3]
        elif self.use_color and not self.use_intensity:
            point_cloud = pc[:, 0:7]
            point_cloud[:, 3:] = point_cloud[:, 3:] / 256.0
        elif not self.use_color and self.use_intensity:
            point_cloud = np.concatenate((pc[:, 0:3], pc[:, 7]), axis=1)
        else:
            point_cloud = pc
            point_cloud[:, 3:7] = point_cloud[:, 3:7] / 256.0

        # ------------------------------- Wireframe ------------------------------
        # load wireframe
        wireframe_file = self.wireframe_files[index]
        wf_vertices, wf_edges = load_wireframe(wireframe_file)

        # ------------------------------- Dataset Preprocessing ------------------------------
        # Use unified preprocessing pipeline with caching
        result = self.preprocessor.process_point_cloud(
                point_cloud=point_cloud,
                wireframe_vertices=wf_vertices,
                point_cloud_file=pc_file,
                wireframe_file=wireframe_file,
                use_cache=True,
                verbose=self.show_preprocessing_details  # Only show details on first epoch
                )

        # ALWAYS use preprocessed data from the pipeline
        point_cloud = result['processed_points']  # Always X Y Z GroupID BorderWeight
        wf_vertices = result['processed_wireframe']  # Always processed vertices
        
        if self.normalize:
            # Only store metadata if normalization was used
            centroid = result['metadata']['centroid']
            max_distance = result['metadata']['max_distance']

        # ------------------------------- Data Augmentation (BEFORE sampling) ------------------------------
        # Apply augmentation AFTER preprocessing to preserve GroupID and BorderWeight
        # Only augment during training, not testing
        if self.augment and self.split_set == 'train':
            # Use epoch for augmentation seed - same augmentation for all samples in an epoch
            # Different augmentation each epoch, but reproducible across runs
            aug_seed = self.epoch % (2**31)
            aug_rng = np.random.RandomState(aug_seed)
            
            # Random Z-axis rotation in multiples of 90° (0°, 90°, 180°, 270°)
            rotation_choice = aug_rng.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
            rot_angle = rotation_choice * np.pi / 2
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            wf_vertices[:, 0:3] = np.dot(wf_vertices[:, 0:3], np.transpose(rot_mat))
            
            # Small Gaussian jittering (σ=0.01 in normalized space)
            jitter_std = 0.01
            point_jitter = aug_rng.normal(0, jitter_std, size=point_cloud[:, 0:3].shape)
            vertex_jitter = aug_rng.normal(0, jitter_std, size=wf_vertices[:, 0:3].shape)
            point_cloud[:, 0:3] += point_jitter
            wf_vertices[:, 0:3] += vertex_jitter
            
            # Note: GroupID (column 3) and BorderWeight (column 4) remain unchanged
            # Note: These augmentations are applied in normalized space

        # ------------------------------- Random Point Sampling ------------------------------
        if self.num_points:
            # Use epoch + sample index for deterministic-but-varying sampling
            # Different points each epoch, but reproducible across runs
            sample_seed = (self.epoch * 10000 + index) % (2**31)  # Combine epoch and index
            # Use Farthest Point Sampling for better coverage
            point_cloud = farthest_point_sampling(point_cloud, self.num_points, seed=sample_seed)

        # -------------------------------Edge Vertices ------------------------
        wf_edges_vertices = np.stack((wf_vertices[wf_edges[:, 0]], wf_vertices[wf_edges[:, 1]]), axis=1)
        wf_edges_vertices = wf_edges_vertices[
            np.arange(wf_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(wf_edges_vertices[:, :, -1]),
                                                                        axis=1)]
        wf_centers = (wf_edges_vertices[..., 0, :] + wf_edges_vertices[..., 1, :]) / 2
        wf_edge_number = wf_edges.shape[0]

        # ------------------------------- Return Dict ------------------------------
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['wf_vertices'] = wf_vertices.astype(np.float32)
        ret_dict['wf_edges'] = wf_edges.astype(np.int64)
        ret_dict['wf_centers'] = wf_centers.astype(np.float32)
        ret_dict['wf_edge_number'] = wf_edge_number
        ret_dict['wf_edges_vertices'] = wf_edges_vertices.reshape((-1, 6)).astype(np.float32)
        if self.normalize:
            ret_dict['centroid'] = centroid
            ret_dict['max_distance'] = max_distance
        ret_dict['scan_idx'] = np.array(os.path.splitext(os.path.basename(pc_file))[0]).astype(np.int64)
        return ret_dict

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
                    max_len = max([len(v) for v in val])
                    wf = np.ones((len(batch), max_len, val[0].shape[-1]), dtype=np.float32) * -1e1
                    for i in range(len(batch)):
                        wf[i, :len(val[i]), :] = val[i]
                    ret_dict[key] = torch.from_numpy(wf)
                else:
                    ret_dict[key] = torch.tensor(np.array(input_dict[key]))
            except:
                print('Error in collate_batch: key=%s' % key)
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
