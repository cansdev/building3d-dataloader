#!/usr/bin/env python3
"""
Evaluation Script for Building3D Corner Detection

This script allows you to:
1. Select a test sample from the dataset
2. Load refined predictions from output/refined_predictions
3. Evaluate predictions against ground truth using AP Calculator
4. Visualize the results to verify metrics

Usage:
    python evaluate.py
"""

import os
import numpy as np
import glob
import torch
import matplotlib.pyplot as plt

from eval.ap_calculator import APCalculator
from datasets import building3d, build_dataset
import yaml
from easydict import EasyDict

# Import visualization functions
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'visualize'))
from visualize.visualize_pointnet2 import (
    visualize_pointnet2_results, 
    setup_environment_non_interactive, 
    load_trained_model
)


def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def get_available_test_samples(test_dir='datasets/demo_dataset/test'):
    """
    Get list of available test samples
    
    Returns:
        List of sample IDs (strings)
    """
    wireframe_dir = os.path.join(test_dir, 'wireframe')
    
    if not os.path.exists(wireframe_dir):
        raise FileNotFoundError(f"Test directory not found: {wireframe_dir}")
    
    # Get all .obj files
    obj_files = glob.glob(os.path.join(wireframe_dir, '*.obj'))
    
    # Extract sample IDs from filenames
    sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in obj_files]
    sample_ids.sort()
    
    return sample_ids


def load_refined_predictions(sample_id, refined_dir='output/refined_predictions'):
    """
    Load refined predictions from npz file
    
    Args:
        sample_id: Sample ID (e.g., '10000')
        refined_dir: Directory containing refined predictions
        
    Returns:
        Dictionary with refined predictions or None if not found
    """
    refined_path = os.path.join(refined_dir, f"{sample_id}_refined.npz")
    
    if not os.path.exists(refined_path):
        return None
    
    try:
        data = np.load(refined_path)
        return {
            'raw_corners': data['raw_corners'],
            'raw_probs': data['raw_probs'],
            'refined_corners': data['refined_corners'],
            'cluster_labels': data['cluster_labels'],
            'cluster_sizes': data['cluster_sizes']
        }
    except Exception as e:
        print(f"Error loading refined predictions: {e}")
        return None


def evaluate_sample(sample_id, test_dir='datasets/demo_dataset/test', 
                   refined_dir='output/refined_predictions', 
                   distance_thresh=1.0,
                   preprocessed_dir='datasets/preprocessed'):
    """
    Evaluate a single test sample
    
    Args:
        sample_id: Sample ID (e.g., '10000')
        test_dir: Directory containing test data
        refined_dir: Directory containing refined predictions
        distance_thresh: Distance threshold for AP calculation
        preprocessed_dir: Directory containing preprocessed/normalized data
    """
    print("\n" + "="*70)
    print(f"EVALUATING SAMPLE: {sample_id}")
    print("="*70)
    
    # Load ground truth
    gt_xyz_path = os.path.join(test_dir, 'xyz', f'{sample_id}.xyz')
    gt_wireframe_path = os.path.join(test_dir, 'wireframe', f'{sample_id}.obj')
    
    if not os.path.exists(gt_xyz_path):
        raise FileNotFoundError(f"Ground truth XYZ file not found: {gt_xyz_path}")
    if not os.path.exists(gt_wireframe_path):
        raise FileNotFoundError(f"Ground truth wireframe file not found: {gt_wireframe_path}")
    
    # Load point cloud (for info only)
    pc = np.loadtxt(gt_xyz_path, dtype=np.float64)
    print(f"✓ Loaded point cloud: {pc.shape[0]} points")
    
    # Load ground truth wireframe (RAW coordinates)
    gt_vertices_raw, gt_edges = building3d.load_wireframe(gt_wireframe_path)
    print(f"✓ Loaded ground truth: {len(gt_vertices_raw)} corners, {len(gt_edges)} edges")
    
    # Load normalization parameters from preprocessed cache
    preprocessed_path = os.path.join(preprocessed_dir, 'test', f'{sample_id}.npz')
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(
            f"Preprocessed data not found: {preprocessed_path}\n"
            f"Please ensure the dataset has been preprocessed by loading it once."
        )
    
    preprocessed = np.load(preprocessed_path)
    centroid = preprocessed['centroid']
    max_distance = preprocessed['max_distance']
    
    # Normalize ground truth vertices to match prediction space
    gt_vertices = (gt_vertices_raw - centroid) / max_distance
    print(f"✓ Normalized ground truth using centroid={centroid}, max_distance={max_distance:.4f}")
    
    # Load refined predictions
    refined_data = load_refined_predictions(sample_id, refined_dir)
    
    if refined_data is None:
        print(f"\n⚠ ERROR: No refined predictions found for sample {sample_id}")
        print(f"  Expected file: {refined_dir}/{sample_id}_refined.npz")
        print(f"  Please run test.py first to generate refined predictions.")
        return None
    
    pd_vertices = refined_data['refined_corners']
    print(f"✓ Loaded refined predictions: {len(pd_vertices)} corners")
    
    # Since refined predictions only contain corners (no edges),
    # we set predicted edges to empty array
    # The AP calculator will handle this case (corner-only evaluation)
    pd_edges = np.empty((0, 2), dtype=np.int32)
    
    # Prepare edge vertices for ground truth
    gt_edges_vertices = np.stack((gt_vertices[gt_edges[:, 0]], gt_vertices[gt_edges[:, 1]]), axis=1)
    gt_edges_vertices = gt_edges_vertices[
        np.arange(gt_edges_vertices.shape[0])[:, np.newaxis], 
        np.flip(np.argsort(gt_edges_vertices[:, :, -1]), axis=1)
    ]
    
    # For predicted edges (empty), create empty array with correct shape
    pd_edges_vertices = np.empty((0, 2, 3), dtype=np.float64)
    
    # Debug: Check distance between predicted and ground truth corners
    if len(pd_vertices) > 0 and len(gt_vertices) > 0:
        from scipy.spatial.distance import cdist
        distances = cdist(pd_vertices, gt_vertices)
        min_distances = np.min(distances, axis=1)
        print(f"\nDistance analysis (predicted to nearest GT corner):")
        print(f"  Min distance: {min_distances.min():.4f}")
        print(f"  Max distance: {min_distances.max():.4f}")
        print(f"  Mean distance: {min_distances.mean():.4f}")
        print(f"  Median distance: {np.median(min_distances):.4f}")
        print(f"  Corners within threshold ({distance_thresh}): {(min_distances <= distance_thresh).sum()}/{len(pd_vertices)}")
    
    # Create batch dictionary for AP calculator
    batch = dict()
    batch['predicted_vertices'] = pd_vertices[np.newaxis, :]
    batch['predicted_edges'] = pd_edges[np.newaxis, :]
    batch['pred_edges_vertices'] = pd_edges_vertices.reshape((1, -1, 2, 3))
    
    batch['wf_vertices'] = gt_vertices[np.newaxis, :]
    batch['wf_edges'] = gt_edges[np.newaxis, :]
    batch['wf_edges_vertices'] = gt_edges_vertices.reshape((1, -1, 2, 3))
    
    # Run AP calculator
    print("\n" + "-"*70)
    print("EVALUATION METRICS")
    print(f"Distance threshold: {distance_thresh}")
    print("-"*70)
    
    ap_calculator = APCalculator(distance_thresh=distance_thresh)
    ap_calculator.compute_metrics(batch)
    
    # Custom output for corner-only evaluation (no edges)
    if len(pd_vertices) > 0 and len(pd_edges) == 0:
        print("\n📊 CORNER-ONLY EVALUATION (No edge predictions)")
        print("-"*70)
        
        # Calculate corner metrics manually
        tp_corners = ap_calculator.ap_dict['tp_corners']
        tp_fp_corners = ap_calculator.ap_dict['tp_fp_corners']
        tp_fn_corners = ap_calculator.ap_dict['tp_fn_corners']
        
        if tp_corners > 0:
            avg_corner_offset = ap_calculator.ap_dict['distance'] / tp_corners
            print(f"Average Corner Offset: {avg_corner_offset:.6f}")
        else:
            print("Average Corner Offset: N/A (no true positives)")
        
        # Corner metrics
        if tp_fp_corners > 0:
            corners_precision = tp_corners / tp_fp_corners
            print(f"Corners Precision: {corners_precision:.6f}")
        else:
            corners_precision = 0.0
            print("Corners Precision: 0.0 (no predictions)")
        
        if tp_fn_corners > 0:
            corners_recall = tp_corners / tp_fn_corners
            print(f"Corners Recall: {corners_recall:.6f}")
        else:
            corners_recall = 0.0
            print("Corners Recall: 0.0 (no ground truth)")
        
        if corners_precision + corners_recall > 0:
            corners_f1 = 2 * corners_precision * corners_recall / (corners_precision + corners_recall)
            print(f"Corners F1: {corners_f1:.6f}")
        else:
            print("Corners F1: 0.0")
        
        # Summary
        print("\n" + "-"*70)
        print("SUMMARY")
        print("-"*70)
        print(f"Predicted corners: {tp_fp_corners}")
        print(f"Ground truth corners: {tp_fn_corners}")
        print(f"True positives: {tp_corners}")
        print(f"False positives: {tp_fp_corners - tp_corners}")
        print(f"False negatives: {tp_fn_corners - tp_corners}")
        
        print("\n⚠ NOTE: Edge metrics not available (no edge predictions)")
        print("   Refined predictions contain only corners, not edges.")
        
    else:
        # Standard output with both corners and edges
        try:
            ap_calculator.output_accuracy()
        except ZeroDivisionError as e:
            print("\n⚠ ERROR: Division by zero in metric calculation")
            print("\nRaw metrics from AP calculator:")
            print(f"  True positive corners: {ap_calculator.ap_dict['tp_corners']}")
            print(f"  Predicted corners (tp+fp): {ap_calculator.ap_dict['tp_fp_corners']}")
            print(f"  Ground truth corners (tp+fn): {ap_calculator.ap_dict['tp_fn_corners']}")
            print(f"  True positive edges: {ap_calculator.ap_dict['tp_edges']}")
            print(f"  Predicted edges (tp+fp): {ap_calculator.ap_dict['tp_fp_edges']}")
            print(f"  Ground truth edges (tp+fn): {ap_calculator.ap_dict['tp_fn_edges']}")
            print("\nThis error occurred despite having predictions. Please check the data.")
            raise
    
    print("-"*70)
    
    return {
        'sample_id': sample_id,
        'pc': pc,
        'gt_vertices': gt_vertices,
        'gt_edges': gt_edges,
        'pd_vertices': pd_vertices,
        'refined_data': refined_data
    }


def visualize_evaluation(sample_id, dataset, model=None, device=None):
    """
    Visualize the evaluated sample using visualize_pointnet2
    
    Args:
        sample_id: Sample ID to visualize
        dataset: Dataset object
        model: Trained model (optional, will load if None)
        device: Device to use (optional)
    """
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    # Find the sample index in the dataset
    sample_idx = None
    for idx in range(len(dataset)):
        if str(int(dataset[idx]['scan_idx'])) == str(sample_id):
            sample_idx = idx
            break
    
    if sample_idx is None:
        print(f"⚠ WARNING: Could not find sample {sample_id} in dataset")
        return
    
    print(f"Visualizing sample {sample_id} (dataset index {sample_idx})...")
    print("The visualization shows:")
    print("  - Left: Ground truth corners")
    print("  - Middle: Raw model predictions (adjust threshold with slider)")
    print("  - Right: DBSCAN refined predictions")
    print("\nClose the visualization window to continue.")
    
    # Visualize using the existing visualization function
    fig = visualize_pointnet2_results(
        model, 
        dataset, 
        sample_idx, 
        device, 
        threshold=0.3,  # Default threshold
        use_refined=True
    )
    plt.show()


def main():
    """Main interactive evaluation function"""
    print("\n" + "="*70)
    print("Building3D Corner Detection - Evaluation Tool")
    print("="*70)
    print("\nThis tool evaluates refined predictions against ground truth and")
    print("displays visualizations to verify the metrics.")
    
    # Get available test samples
    try:
        sample_ids = get_available_test_samples()
    except FileNotFoundError as e:
        print(f"\n⚠ ERROR: {e}")
        return
    
    print(f"\n✓ Found {len(sample_ids)} test samples")
    print("\nAvailable samples:")
    for i, sample_id in enumerate(sample_ids):
        print(f"  {i}: {sample_id}")
    
    # Setup environment for visualization
    print("\nLoading model and dataset for visualization...")
    try:
        env = setup_environment_non_interactive()
        model = load_trained_model(env['model_path'], env['input_channels'], env['device'])
        test_dataset = env['datasets']['test']
        print(f"✓ Model loaded from: {env['model_path']}")
        print(f"✓ Test dataset loaded: {len(test_dataset)} samples")
    except Exception as e:
        print(f"\n⚠ WARNING: Could not load model/dataset for visualization: {e}")
        print("  Evaluation will run but visualization will be skipped.")
        model = None
        test_dataset = None
        env = None
    
    # Default distance threshold
    distance_thresh = 1.0
    
    # Interactive sample selection loop
    while True:
        print("\n" + "="*70)
        print(f"Current distance threshold: {distance_thresh}")
        print(f"\nSelect sample (0-{len(sample_ids)-1}), 't' to change threshold, or 'q' to quit: ")
        choice = input("Choice: ").strip()
        
        if choice.lower() == 'q':
            print("\nExiting evaluation tool. Goodbye!")
            break
        
        if choice.lower() == 't':
            try:
                new_thresh = input(f"Enter new distance threshold (current: {distance_thresh}): ").strip()
                distance_thresh = float(new_thresh)
                print(f"✓ Distance threshold set to {distance_thresh}")
            except ValueError:
                print("⚠ Invalid number, keeping current threshold")
            continue
        
        try:
            sample_idx_choice = int(choice)
            
            if 0 <= sample_idx_choice < len(sample_ids):
                sample_id = sample_ids[sample_idx_choice]
                
                # Evaluate the sample with current threshold
                result = evaluate_sample(sample_id, distance_thresh=distance_thresh)
                
                if result is not None and model is not None and test_dataset is not None:
                    # Visualize the results
                    visualize_evaluation(sample_id, test_dataset, model, env['device'])
                elif result is not None:
                    print("\n⚠ Skipping visualization (model/dataset not loaded)")
            else:
                print(f"⚠ Please enter a number between 0 and {len(sample_ids)-1}")
        
        except ValueError:
            print("⚠ Please enter a valid number, 't' to change threshold, or 'q' to quit")
        except KeyboardInterrupt:
            print("\n\nExiting evaluation tool. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

