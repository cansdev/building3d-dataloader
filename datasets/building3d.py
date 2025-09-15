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
from utils.normalization import normalize_data
from utils.surface_grouping import surface_aware_normalization_pipeline

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


class Building3DReconstructionDataset(Dataset):
    def __init__(self, dataset_config, split_set, logger=None):
        self.dataset_config = dataset_config
        self.roof_dir = dataset_config.root_dir
        self.num_points = dataset_config.num_points
        self.use_color = dataset_config.use_color
        self.use_intensity = dataset_config.use_intensity
        self.normalize = dataset_config.normalize
        self.augment = dataset_config.augment
        self.use_surface_grouping = getattr(dataset_config, 'use_surface_grouping', False)
        self.use_outlier_removal = getattr(dataset_config, 'use_outlier_removal', True)

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
        if self.normalize:
            if self.use_surface_grouping:
                # Use advanced surface-aware processing pipeline
                print(f"Using surface-aware processing for sample {index}")
                
                # Configure pipeline parameters (more conservative outlier removal)
                outlier_params = {
                    'remove_statistical': self.use_outlier_removal,
                    'remove_radius': self.use_outlier_removal,
                    'remove_elevation': self.use_outlier_removal,
                    'stat_k': 15,
                    'stat_std_ratio': 3.0,  # More conservative (was 2.0)
                    'radius_threshold': 0.04,
                    'radius_min_neighbors': 2,  # More lenient (was 3)
                    'elev_std_ratio': 3.0,  # More conservative (was 2.0)
                    'auto_scale_radii': True
                }
                
                grouping_params = {
                    'coarse_params': {
                        'eps': 0.05,  # Simple spatial clustering parameter
                        'min_samples': 5,
                        'auto_scale_eps': True
                    },
                    'refinement_params': {
                        'merge_distance': 0.05,
                        'min_group_size': 15  # Minimum 15 points per cluster
                    }
                }
                
                processing_params = {
                    'apply_outlier_removal': self.use_outlier_removal,
                    'compute_normals': True,
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
                    normals = surface_results['normals']
                    group_ids = surface_results.get('group_ids', None)
                    
                    # Normalize coordinates to unit scale
                    centroid = np.mean(cleaned_points, axis=0)
                    centered_points = cleaned_points - centroid
                    max_distance = np.max(np.linalg.norm(centered_points, axis=1))
                    
                    if max_distance > 1e-6:
                        normalized_coords = centered_points / max_distance
                    else:
                        normalized_coords = centered_points
                    
                    # Apply same transformation to wireframe
                    wf_vertices = (wf_vertices - centroid) / max_distance
                    
                    # Reconstruct full point cloud with surface information
                    n_cleaned = len(cleaned_points)
                    
                    if point_cloud.shape[1] >= 8:
                        # Preserve original features for cleaned points
                        feature_dim = point_cloud.shape[1]
                        reconstructed_pc = np.zeros((n_cleaned, feature_dim + 4))  # +3 for normals, +1 for group id
                        
                        # Use cleaned coordinates  
                        reconstructed_pc[:, 0:3] = normalized_coords
                        
                        # For features, use nearest neighbor mapping from original to cleaned points
                        if n_cleaned > 0:
                            from sklearn.neighbors import NearestNeighbors
                            original_coords = point_cloud[:, 0:3]
                            nbrs = NearestNeighbors(n_neighbors=1).fit(original_coords)
                            _, nearest_indices = nbrs.kneighbors(cleaned_points)
                            nearest_indices = nearest_indices.flatten()
                            
                            # Copy features from nearest original points
                            reconstructed_pc[:, 3:feature_dim] = point_cloud[nearest_indices, 3:]
                        
                        # Add normals and group ids
                        if normals is not None:
                            reconstructed_pc[:, feature_dim:feature_dim+3] = normals
                        if group_ids is not None:
                            reconstructed_pc[:, feature_dim+3] = group_ids
                        
                        point_cloud = reconstructed_pc
                    else:
                        # Simple case: just coordinates + normals + group ids
                        point_cloud = np.column_stack([
                            normalized_coords,
                            normals if normals is not None else np.zeros((n_cleaned, 3)),
                            group_ids if group_ids is not None else np.zeros(n_cleaned)
                        ])
                else:
                    print("Surface-aware processing failed, falling back to standard normalization")
                    # Fallback to standard normalization
                    point_cloud, wf_vertices, centroid, max_distance, normals = normalize_data(
                        point_cloud, wf_vertices, compute_normals=True, k_neighbors=10,
                        clean_outliers=self.use_outlier_removal
                    )
            else:
                # Standard normalization pipeline
                # Custom outlier removal parameters for better normal quality
                outlier_params = {
                    'stat_k': 15,
                    'stat_std_ratio': 1.5,  # More aggressive statistical outlier removal
                    'radius_threshold': 0.03,  # Smaller radius for tighter clustering
                    'radius_min_neighbors': 3,  # Require more neighbors for validity
                    'elev_std_ratio': 2.0,  # More aggressive elevation filtering
                    'elev_percentiles': (10, 90)  # Tighter percentile range for robust stats
                }
                
                point_cloud, wf_vertices, centroid, max_distance, normals = normalize_data(
                    point_cloud, wf_vertices, compute_normals=True, k_neighbors=10,
                    clean_outliers=self.use_outlier_removal, outlier_params=outlier_params
                )

        if self.num_points:
            point_cloud = random_sampling(point_cloud, self.num_points)

        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                wf_vertices[:, 0] = -1 * wf_vertices[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                wf_vertices[:, 1] = -1 * wf_vertices[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            wf_vertices[:, 0:3] = np.dot(wf_vertices[:, 0:3], np.transpose(rot_mat))

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
