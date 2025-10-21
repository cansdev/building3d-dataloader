"""
Test script for PointNet2 corner detection with DBSCAN clustering refinement.

This script:
1. Loads a trained PointNet2 model
2. Runs inference on the test dataset
3. Applies DBSCAN clustering to refine predictions (group nearby predicted corners)
4. Computes cluster centroids as final refined predictions
5. Saves results for visualization

Usage:
    python test.py --model_path output/corner_detection_model.pth --eps 0.05
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.PointNet2 import PointNet2CornerDetection
import os
import numpy as np
import yaml
from easydict import EasyDict
from datasets import build_dataset
from sklearn.cluster import DBSCAN
import argparse


def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def load_trained_model(model_path, device='cuda'):
    """
    Load a trained PointNet2 model from checkpoint
    
    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        model: Loaded PointNet2 model
        config: Model configuration dictionary
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract config and create model
    model_config = checkpoint['config']
    model = PointNet2CornerDetection(config=model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Model configuration: {model_config}")
    
    return model, model_config


def predict_corners(model, point_cloud, threshold=0.5, device='cuda'):
    """
    Predict corner points from a point cloud using trained model
    
    Args:
        model: Trained PointNet2 model
        point_cloud: Point cloud tensor [N, C]
        threshold: Probability threshold for corner detection
        device: Device to run inference on
    
    Returns:
        corner_points: Predicted corner coordinates [M, 3]
        corner_probs: Corner probabilities [M]
        corner_indices: Original point cloud indices [M]
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        # Add batch dimension if needed
        if point_cloud.dim() == 2:
            point_cloud = point_cloud.unsqueeze(0)  # [1, N, C]
        
        point_cloud = point_cloud.to(device)
        
        # Forward pass
        corner_logits, _ = model(point_cloud)  # [B, N]
        corner_probs = torch.sigmoid(corner_logits)  # [B, N]
        
        # Remove batch dimension
        corner_probs = corner_probs.squeeze(0)  # [N]
        
        # Get points above threshold
        corner_mask = corner_probs > threshold
        corner_indices = torch.where(corner_mask)[0]
        
        # Extract corner coordinates (first 3 dimensions are XYZ)
        corner_points = point_cloud.squeeze(0)[corner_indices, :3]  # [M, 3]
        corner_probs_filtered = corner_probs[corner_indices]  # [M]
    
    return corner_points.cpu().numpy(), corner_probs_filtered.cpu().numpy(), corner_indices.cpu().numpy()


def cluster_corners_dbscan(corner_points, corner_probs, eps=0.05, min_samples=1):
    """
    Cluster predicted corners using DBSCAN and compute cluster centroids
    
    Args:
        corner_points: Predicted corner coordinates [M, 3]
        corner_probs: Corner probabilities [M]
        eps: DBSCAN epsilon parameter (maximum distance between points in a cluster)
        min_samples: Minimum number of samples in a cluster
    
    Returns:
        refined_corners: Cluster centroids as refined corner predictions [K, 3]
        cluster_labels: Cluster label for each input point [M]
        cluster_sizes: Number of points in each cluster [K]
    """
    if len(corner_points) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clustering.fit_predict(corner_points)
    
    # Get unique cluster IDs (excluding noise points labeled as -1)
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Remove noise label
    
    refined_corners = []
    cluster_sizes = []
    
    for label in unique_labels:
        # Get all points in this cluster
        cluster_mask = cluster_labels == label
        cluster_points = corner_points[cluster_mask]
        cluster_point_probs = corner_probs[cluster_mask]
        
        # Compute weighted centroid (weighted by corner probability)
        weights = cluster_point_probs / cluster_point_probs.sum()
        centroid = np.average(cluster_points, axis=0, weights=weights)
        
        refined_corners.append(centroid)
        cluster_sizes.append(len(cluster_points))
    
    refined_corners = np.array(refined_corners)
    cluster_sizes = np.array(cluster_sizes)
    
    # Handle noise points (labeled as -1) - add them as individual corners
    noise_mask = cluster_labels == -1
    if noise_mask.any():
        noise_points = corner_points[noise_mask]
        refined_corners = np.concatenate([refined_corners, noise_points], axis=0)
        cluster_sizes = np.concatenate([cluster_sizes, np.ones(len(noise_points), dtype=int)])
    
    return refined_corners, cluster_labels, cluster_sizes


def refine_predictions_with_clustering(model, data_loader, corner_threshold=0.5, 
                                       dbscan_eps=0.05, dbscan_min_samples=1, 
                                       device='cuda'):
    """
    Run inference on dataset and refine predictions using DBSCAN clustering
    
    Args:
        model: Trained PointNet2 model
        data_loader: DataLoader for inference
        corner_threshold: Probability threshold for corner detection
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        device: Device to run inference on
    
    Returns:
        results: Dictionary containing refined predictions for each sample
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = {}
    total_raw_corners = 0
    total_refined_corners = 0
    total_clusters = 0
    
    print("\n" + "="*60)
    print("Running inference with DBSCAN clustering refinement...")
    print("="*60)
    print(f"Corner threshold: {corner_threshold}")
    print(f"DBSCAN eps: {dbscan_eps}")
    print(f"DBSCAN min_samples: {dbscan_min_samples}")
    print()
    
    for batch_idx, batch in enumerate(data_loader):
        point_clouds = batch['point_clouds'].to(device)  # [B, N, C]
        scan_indices = batch['scan_idx'].cpu().numpy()
        
        # Process each sample in the batch
        for i in range(point_clouds.shape[0]):
            point_cloud = point_clouds[i]  # [N, C]
            scan_idx = scan_indices[i]
            
            # Step 1: Predict corners
            corner_points, corner_probs, corner_indices = predict_corners(
                model, point_cloud, threshold=corner_threshold, device=device
            )
            
            # Step 2: Apply DBSCAN clustering
            refined_corners, cluster_labels, cluster_sizes = cluster_corners_dbscan(
                corner_points, corner_probs, eps=dbscan_eps, min_samples=dbscan_min_samples
            )
            
            # Store results
            results[scan_idx] = {
                'raw_corners': corner_points,
                'raw_probs': corner_probs,
                'raw_indices': corner_indices,
                'refined_corners': refined_corners,
                'cluster_labels': cluster_labels,
                'cluster_sizes': cluster_sizes,
                'num_raw': len(corner_points),
                'num_refined': len(refined_corners),
                'num_clusters': len(np.unique(cluster_labels[cluster_labels != -1]))
            }
            
            total_raw_corners += len(corner_points)
            total_refined_corners += len(refined_corners)
            total_clusters += results[scan_idx]['num_clusters']
            
            # Print sample statistics
            if batch_idx % 5 == 0 and i == 0:
                print(f"Sample {scan_idx}: "
                      f"{len(corner_points)} raw corners -> "
                      f"{len(refined_corners)} refined corners "
                      f"({results[scan_idx]['num_clusters']} clusters)")
    
    # Print summary statistics
    num_samples = len(results)
    print()
    print("="*60)
    print("Clustering Refinement Summary")
    print("="*60)
    print(f"Total samples: {num_samples}")
    print(f"Total raw corners: {total_raw_corners} (avg: {total_raw_corners/num_samples:.1f} per sample)")
    print(f"Total refined corners: {total_refined_corners} (avg: {total_refined_corners/num_samples:.1f} per sample)")
    print(f"Total clusters: {total_clusters} (avg: {total_clusters/num_samples:.1f} per sample)")
    print(f"Reduction ratio: {total_refined_corners/total_raw_corners:.2%}")
    print("="*60)
    
    return results


def save_refined_predictions(results, output_dir='output/refined_predictions'):
    """
    Save refined corner predictions to disk for visualization
    
    Args:
        results: Dictionary containing refined predictions for each sample
        output_dir: Directory to save predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for scan_idx, result in results.items():
        output_file = os.path.join(output_dir, f"{scan_idx}_refined.npz")
        np.savez(
            output_file,
            raw_corners=result['raw_corners'],
            raw_probs=result['raw_probs'],
            refined_corners=result['refined_corners'],
            cluster_labels=result['cluster_labels'],
            cluster_sizes=result['cluster_sizes']
        )
    
    print(f"\nRefined predictions saved to {output_dir}/")
    print(f"Total files saved: {len(results)}")


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Test PointNet2 with DBSCAN clustering refinement')
    parser.add_argument('--model_path', type=str, default='output/corner_detection_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--config_path', type=str, default='datasets/dataset_config.yaml',
                        help='Path to dataset configuration YAML file')
    parser.add_argument('--output_dir', type=str, default='output/refined_predictions',
                        help='Directory to save refined predictions')
    parser.add_argument('--corner_threshold', type=float, default=0.3,
                        help='Probability threshold for corner detection (default: 0.3)')
    parser.add_argument('--dbscan_eps', type=float, default=0.05,
                        help='DBSCAN epsilon parameter - max distance between points in cluster (default: 0.05)')
    parser.add_argument('--dbscan_min_samples', type=int, default=1,
                        help='DBSCAN min_samples - minimum points to form cluster (default: 1)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for inference (default: 4)')
    args = parser.parse_args()
    
    print("="*60)
    print("PointNet2 Corner Detection - Test with DBSCAN Refinement")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Config path: {args.config_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Corner threshold: {args.corner_threshold}")
    print(f"DBSCAN eps: {args.dbscan_eps}")
    print(f"DBSCAN min_samples: {args.dbscan_min_samples}")
    print()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found!")
        print("Please train the model first using train.py")
        return
    
    # Load dataset configuration
    config_path = os.path.join(os.path.dirname(__file__), args.config_path)
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found!")
        return
    
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    model, model_config = load_trained_model(args.model_path, device=device)
    
    # Build test dataset
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Create test data loader
    test_loader = DataLoader(
        building3D_dataset['test'], 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=building3D_dataset['test'].collate_batch,
        pin_memory=True,
        num_workers=2
    )
    
    # Run inference with DBSCAN clustering
    results = refine_predictions_with_clustering(
        model=model,
        data_loader=test_loader,
        corner_threshold=args.corner_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        device=device
    )
    
    # Save refined predictions
    save_refined_predictions(results, output_dir=args.output_dir)
    
    print("\n" + "="*60)
    print("Testing completed successfully!")
    print(f"Use visualize_pointnet2.py to visualize the refined predictions")
    print("="*60)


if __name__ == "__main__":
    main()

