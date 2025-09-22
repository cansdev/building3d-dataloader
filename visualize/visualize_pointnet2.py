#!/usr/bin/env python3
"""Minimal PointNet2 Corner Detection Visualization.

Loads a trained model, runs inference on one sample, and renders a single 3D view
with point cloud, ground-truth corners, and predicted corners (via threshold).
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import build_dataset
from models.PointNet2 import PointNet2CornerDetection
import yaml
from easydict import EasyDict

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        return EasyDict(yaml.load(f, Loader=yaml.FullLoader))

def load_trained_model(model_path, input_channels, device):
    model = PointNet2CornerDetection(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def prepare_inputs(sample, device):
    point_clouds = sample['point_clouds']
    wf_vertices = sample['wf_vertices']
    if isinstance(point_clouds, np.ndarray):
        point_clouds = torch.from_numpy(point_clouds)
    if isinstance(wf_vertices, np.ndarray):
        wf_vertices = torch.from_numpy(wf_vertices)
    return point_clouds.unsqueeze(0).to(device), wf_vertices.unsqueeze(0).to(device)

def visualize_pointnet2_results(model, dataset, sample_idx, device, threshold=0.5):
    sample = dataset[sample_idx]
    point_clouds, wf_vertices = prepare_inputs(sample, device)

    valid_mask = wf_vertices[0, :, 0] > -1.0
    gt_corners = wf_vertices[0, valid_mask].cpu().numpy()

    with torch.no_grad():
        corner_logits, _ = model(point_clouds)
        corner_probs = torch.sigmoid(corner_logits[0])
        corner_predictions = (corner_probs > threshold).cpu().numpy()

    pc_xyz = point_clouds[0, :, :3].cpu().numpy()
    predicted_corners = pc_xyz[corner_predictions]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc_xyz[:, 0], pc_xyz[:, 1], pc_xyz[:, 2], c='lightblue', s=1, alpha=0.4)
    if len(gt_corners) > 0:
        ax.scatter(gt_corners[:, 0], gt_corners[:, 1], gt_corners[:, 2], c='red', s=30, marker='o')
    if len(predicted_corners) > 0:
        ax.scatter(predicted_corners[:, 0], predicted_corners[:, 1], predicted_corners[:, 2], c='orange', s=50, marker='s', edgecolors='black', linewidth=0.5)
    ax.set_title(f'PointNet2 (threshold={threshold})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Minimal probability heatmap (reference scale) on the right
    # Reserve a bit of space on the right for the heatmap
    plt.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])
    heat_ax = fig.add_axes([0.96, 0.15, 0.02, 0.7])
    grad = np.linspace(0.0, 1.0, 256)[:, None]
    heat_ax.imshow(grad, aspect='auto', cmap='viridis', origin='lower', extent=(0, 1, 0, 1))
    # Indicate threshold
    heat_ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1)
    heat_ax.set_yticks([0.0, 0.5, 1.0])
    heat_ax.set_yticklabels(['0.0', '0.5', '1.0'])
    heat_ax.set_xticks([])
    heat_ax.set_title('prob', fontsize=9, pad=4)
    return fig

def compute_input_channels(cfg):
    channels = 3
    if cfg.Building3D.use_color:
        channels += 4
    if cfg.Building3D.use_intensity:
        channels += 1
    return channels

def load_resources(split):
    cfg = cfg_from_yaml_file('../datasets/dataset_config.yaml')
    if cfg.Building3D.root_dir.startswith('./'):
        cfg.Building3D.root_dir = '../' + cfg.Building3D.root_dir[2:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = '../output/corner-detection-original.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model file not found at {model_path}')
    input_channels = compute_input_channels(cfg)
    model = load_trained_model(model_path, input_channels, device)
    ds = build_dataset(cfg.Building3D)[split]
    if len(ds) == 0:
        raise RuntimeError(f'No samples found in the {split} dataset')
    return model, ds, device

def main():
    # Example usage: python visualize/visualize_pointnet2.py --sample 0 --split test --threshold 0.5
    parser = argparse.ArgumentParser(description='Visualize PointNet2 corner detection results')
    parser.add_argument('--sample', type=int, default=0, help='Sample index to visualize (default: 0)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for corner predictions')
    parser.add_argument('--split', type=str, choices=['train', 'test'], default='test', help='Dataset split to use')
    args = parser.parse_args()

    model, dataset, device = load_resources(args.split)
    if args.sample < 0 or args.sample >= len(dataset):
        raise IndexError(f'Sample index {args.sample} out of range (0-{len(dataset)-1})')
    fig = visualize_pointnet2_results(model, dataset, args.sample, device, args.threshold)
    plt.show()

if __name__ == "__main__":
    main()