#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# Training with DGCNN model and CUDA support

import torch
import torch.nn as nn
import numpy as np
from utils.cuda_utils import to_cuda, print_gpu_memory
from models.dgcnn_model import BasicDGCNN

def main_training_setup():
    """Setup training configuration"""
    print("Training setup initialized with DGCNN")
    return True

def train_on_real_dataset(train_loader, device):
    """
    Train DGCNN model on vertex coordinate prediction with CUDA support.
    
    Args:
        train_loader: DataLoader for training data
        device: Device to use (cuda or cpu)
    
    Returns:
        dict: Training results
    """
    print(f"Training DGCNN for vertex coordinate prediction with {device}")
    
    # Initialize DGCNN model for 5 features: X Y Z GroupID BorderWeight
    model = BasicDGCNN(input_dim=5, k=20, max_vertices=64)
    model = model.to(device)
    
    # Optimizer - FIXED LR for overfitting (no scheduler that kills learning!)
    # LOWERED LR for stability with single sample (0.001 → 0.0001 based on analysis)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # Reduced LR and weight decay
    
    # Improved loss functions
    # Use Huber loss (SmoothL1) to prevent gradient explosion from large initial errors
    # Delta=0.1 means: use L2 for errors < 0.1m, L1 for errors > 0.1m
    coord_loss_fn = nn.HuberLoss(delta=0.1)  # Robust to outliers, strong gradients for small errors
    existence_loss_fn = nn.BCEWithLogitsLoss()  # Per-vertex existence loss (better than global count!)
    
    print(f"DGCNN model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # MULTI-SAMPLE TRAINING: Train on 4 samples for better generalization
    target_samples = [1, 10, 100, 1003]  # Four samples!
    training_batch = None
    
    # Collect samples from multiple batches
    collected_samples = {}  # Dict to store {sample_id: batch_data}
    
    for batch_idx, batch_data in enumerate(train_loader):
        batch_on_device = to_cuda(batch_data, device)
        
        # Check if this batch contains our target samples
        if 'scan_idx' in batch_on_device:
            scan_indices = batch_on_device['scan_idx']
            
            # Find which of our target samples are in this batch
            for target_sample in target_samples:
                if target_sample in scan_indices and target_sample not in collected_samples:
                    # Store this sample with its batch data
                    collected_samples[target_sample] = batch_on_device
                    print(f"  Found sample {target_sample} in batch {batch_idx + 1} ({len(collected_samples)}/4)")
                    
            # Stop if we have all samples
            if len(collected_samples) >= 4:
                break
    
    if len(collected_samples) < 4:
        print(f"❌ Could not find all 4 samples! Found: {list(collected_samples.keys())}")
        return {'error': f'Only found {len(collected_samples)} samples: {list(collected_samples.keys())}'}
    
    # Now build the training batch from collected samples
    print(f"✅ Found all 4 samples: {list(collected_samples.keys())}")
    
    # Extract and stack all samples in order
    training_batch = {}
    
    # For each key, extract from each sample's batch and stack
    first_sample_batch = list(collected_samples.values())[0]
    for key in first_sample_batch.keys():
        if isinstance(first_sample_batch[key], torch.Tensor) and len(first_sample_batch[key].shape) > 0:
            # Find position of each target sample in its batch and extract
            samples_list = []
            for target_sample in target_samples:
                batch_data = collected_samples[target_sample]
                scan_indices = batch_data['scan_idx']
                sample_pos = (scan_indices == target_sample).nonzero(as_tuple=True)[0][0]
                samples_list.append(batch_data[key][sample_pos:sample_pos+1])
            
            # Handle padding size mismatch for all wireframe-related keys
            if key in ['wf_vertices', 'wf_edges', 'wf_centers', 'wf_edges_vertices']:
                # Find the maximum size across all samples
                max_size = max(s.shape[1] for s in samples_list)
                # Pad all samples to max_size
                padded_samples = []
                for sample in samples_list:
                    if sample.shape[1] < max_size:
                        # Pad with appropriate sentinel value
                        pad_size = max_size - sample.shape[1]
                        if key == 'wf_vertices' or key == 'wf_centers' or key == 'wf_edges_vertices':
                            # Pad with -10 for coordinates
                            padding = torch.ones(1, pad_size, *sample.shape[2:], device=sample.device) * -1e1
                        elif key == 'wf_edges':
                            # Pad with -1 for edges
                            padding = torch.ones(1, pad_size, *sample.shape[2:], device=sample.device) * -1
                        sample = torch.cat([sample, padding], dim=1)
                    padded_samples.append(sample)
                training_batch[key] = torch.cat(padded_samples, dim=0)  # [12, max_size, ...]
            else:
                training_batch[key] = torch.cat(samples_list, dim=0)  # [12, ...]
        else:
            training_batch[key] = first_sample_batch[key]
    
    # Extract data
    points = training_batch['point_clouds']  # [4, N, 5] - Four samples!
    vertices_gt_original = training_batch['wf_vertices']  # [4, V_padded, 3] ground truth vertex coordinates (may contain -10 padding)
    
    print(f"Training on FOUR SAMPLES: {list(collected_samples.keys())}")
    print(f"   Point clouds: {points.shape}")
    print(f"   GT vertices (with padding): {vertices_gt_original.shape}")
    
    # ===== OPTION 2: Transform GT to canonical space =====
    # We need to align GT to the same canonical space as predictions
    from models.dgcnn_model import compute_pca_alignment
    coords = points[:, :, :3]  # Extract XYZ
    with torch.no_grad():
        _, _, rotation_matrix, centroid = compute_pca_alignment(coords, return_transform=True)
    
    # Transform GT vertices to canonical space
    # Remove padding first to avoid transforming invalid vertices
    B = vertices_gt_original.shape[0]
    vertices_gt_canonical_list = []
    for b in range(B):
        gt_b = vertices_gt_original[b]  # [V_padded, 3]
        valid_mask = gt_b[:, 0] > -9.0  # Filter padding (-10.0)
        gt_valid = gt_b[valid_mask]  # [V_real, 3]
        
        # Transform to canonical: (gt - centroid) @ rotation_matrix
        gt_centered = gt_valid - centroid[b]  # [V_real, 3]
        gt_canonical = torch.matmul(gt_centered, rotation_matrix[b])  # [V_real, 3]
        
        # Pad back to original shape
        gt_padded = torch.full_like(gt_b, -10.0)
        gt_padded[:gt_canonical.shape[0]] = gt_canonical
        vertices_gt_canonical_list.append(gt_padded)
    
    vertices_gt = torch.stack(vertices_gt_canonical_list, dim=0)  # Now in canonical space!
    print(f"   GT transformed to canonical space for comparison")
    
    # Extract normalization scale factor for REAL error calculation
    # This is CRITICAL - without it, errors are reported in normalized space!
    max_distance = None
    if 'max_distance' in training_batch:
        max_distance = training_batch['max_distance'][0].item()  # Convert tensor to float
        print(f"   Normalization scale factor: {max_distance:.2f}m")
        print(f"   (Errors will be converted: normalized_error × {max_distance:.2f} = real_error)")
    else:
        print(f"   WARNING: No normalization metadata found! Errors reported in raw coordinates.")
    
    # CRITICAL FIX: Filter out padded vertices (padded with -10 by collate_batch)
    # Process EACH sample in the batch separately
    B = points.shape[0]  # Batch size = 12
    valid_vertices_list = []
    num_vertices_list = []
    
    for i in range(B):
        # Identify valid vertices for this sample (not padded with -10)
        valid_mask_i = (vertices_gt[i, :, 0] > -9.0)  # Padding is -10, so valid vertices are > -9
        valid_vertices_i = vertices_gt[i, valid_mask_i, :]  # [V_i, 3]
        V_i = valid_vertices_i.shape[0]  # Actual number of vertices for sample i
        
        valid_vertices_list.append(valid_vertices_i)
        num_vertices_list.append(V_i)
        
        # Only print details for first 2 and last sample to reduce clutter
        if i < 2 or i == B - 1:
            print(f"   Sample {i+1}: ✅ Filtered padding: {V_i} REAL vertices (removed {vertices_gt.shape[1] - V_i} padded)")
            print(f"     Vertex coordinate ranges: X=[{valid_vertices_i[:, 0].min().item():.3f}, {valid_vertices_i[:, 0].max().item():.3f}] Y=[{valid_vertices_i[:, 1].min().item():.3f}, {valid_vertices_i[:, 1].max().item():.3f}] Z=[{valid_vertices_i[:, 2].min().item():.3f}, {valid_vertices_i[:, 2].max().item():.3f}]")
        elif i == 2:
            print(f"   ... (samples 3-{B} details omitted for brevity) ...")
    
    # Prepare ground truth - properly padded for batch processing
    max_vertices = model.max_vertices
    num_vertices_gt = torch.tensor(num_vertices_list, dtype=torch.float32, device=device)  # [B]
    
    # Create properly padded ground truth tensor (pad with ZEROS, not -10)
    vertices_gt_padded = torch.zeros(B, max_vertices, 3, device=device)
    for i in range(B):
        V_i = num_vertices_list[i]
        if V_i > max_vertices:
            print(f"   Warning: Sample {i+1} has {V_i} vertices, but model max is {max_vertices}. Truncating.")
            V_i = max_vertices
            num_vertices_gt[i] = max_vertices
        vertices_gt_padded[i, :V_i, :] = valid_vertices_list[i][:V_i, :]  # Only copy real vertices
    
    # Extract feature information
    points_xyz = points[:, :, :3]      # [B, N, 3] - XYZ coordinates
    group_ids = points[:, :, 3:4]      # [B, N, 1] - Group IDs
    border_weights = points[:, :, 4:5] # [B, N, 1] - Border weights
    
    print(f"   Feature analysis:")
    print(f"   Point cloud shape: {points.shape}")
    print(f"   Group IDs range: {group_ids.min():.1f} to {group_ids.max():.1f}")
    print(f"   Border weights range: {border_weights.min():.3f} to {border_weights.max():.3f}")
    print(f"   Ground truth vertices: {num_vertices_list}")
    
    # Training loop for OVERFITTING ON SINGLE SAMPLE
    model.train()
    num_epochs = 2000  # More epochs needed with lower LR for stable convergence
    
    print(f"Starting OVERFITTING training for {num_epochs} epochs...")
    print(f"Goal: Achieve < 10mm real-world error on SINGLE sample (perfect overfit)")
    print(f"Strategy: Low LR (0.0001) + Long training (5000 epochs) for stable convergence")
    if max_distance:
        print(f"Target: < 0.01m real = < {0.01/max_distance:.4f} normalized error")
    
    best_loss = float('inf')
    best_vertex_error = float('inf')
    
    # Adaptive loss weighting - Aggressive for single sample
    # Single sample can handle much stronger coordinate focus
    base_coord_weight = 50.0  # Higher for single sample (was 30.0)
    base_existence_weight = 1.0  # Existence prediction weight
    
    for epoch in range(num_epochs):
        # Set epoch for deterministic-but-varying random sampling
        # This ensures different point subsets each epoch while maintaining reproducibility
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(points)
        pred_vertices = outputs['vertex_coords']        # [B, max_vertices, 3]
        existence_logits = outputs['existence_logits']  # [B, max_vertices] - raw logits
        existence_probs = outputs['existence_probs']    # [B, max_vertices] - sigmoid probabilities
        pred_num_vertices = outputs['num_vertices']     # [B] - dynamic count from threshold
        
        # Create per-vertex existence targets (binary mask)
        # For each sample, first N vertices should exist (1.0), rest should not (0.0)
        existence_target = torch.zeros(B, model.max_vertices).to(device)
        for i in range(B):
            V_i = num_vertices_list[i]
            existence_target[i, :V_i] = 1.0  # First V_i vertices exist
        
        # Per-vertex existence loss - Each vertex gets its own supervision signal!
        # This is much better than a single global count loss
        existence_loss = existence_loss_fn(existence_logits, existence_target)
        
        # Coordinate loss - only for existing vertices (masked by ground truth)
        coord_losses = []
        for i in range(B):
            V_i = num_vertices_list[i]
            # Only compute loss for real vertices (not padding)
            coord_loss_i = coord_loss_fn(
                pred_vertices[i:i+1, :V_i, :], 
                vertices_gt_padded[i:i+1, :V_i, :]
            )
            coord_losses.append(coord_loss_i)
        coord_loss = torch.stack(coord_losses).mean()
        
        # Adaptive loss weighting - MODERATE for single sample (50→100 for stability)
        progress = epoch / num_epochs
        # Coordinates: Start at 50.0, increase to 100.0 (less aggressive for stability)
        coord_weight = base_coord_weight * (1.0 + 1.0 * progress)  # 50.0 → 100.0
        # Existence: Start at 1.0, decrease to 0.5 (gentler)
        existence_weight = base_existence_weight * (1.0 - 0.5 * progress)  # 1.0 → 0.5
        
        # Combined loss with adaptive weighting
        total_loss = coord_weight * coord_loss + existence_weight * existence_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent explosion with stable norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Standard clipping for stability
        
        optimizer.step()
        
        # No learning rate scheduler - using fixed LR for overfitting!
        
        # Track best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        if (epoch + 1) % 5 == 0 or epoch < 20:
            with torch.no_grad():
                # Compute metrics for each sample
                sample_metrics = []
                for i in range(B):
                    V_i = num_vertices_list[i]
                    
                    # Get predicted vertex count from existence probabilities
                    pred_num_i = pred_num_vertices[i].item()
                    
                    # Calculate how many vertices the model thinks exist (prob > 0.5)
                    exists_mask = existence_probs[i] > 0.5
                    num_exists = exists_mask.sum().item()
                    
                    # Mean existence probability for real vs fake vertices
                    real_exist_prob = existence_probs[i, :V_i].mean().item()
                    fake_exist_prob = existence_probs[i, V_i:].mean().item() if V_i < model.max_vertices else 0.0
                    
                    vertex_dist_i = torch.norm(pred_vertices[i:i+1, :V_i, :] - vertices_gt_padded[i:i+1, :V_i, :], dim=2)
                    mean_error_norm = vertex_dist_i.mean().item()  # Normalized space error
                    
                    # CRITICAL FIX: Convert to REAL-WORLD error in meters
                    if max_distance is not None:
                        mean_error_world = mean_error_norm * max_distance
                        # Use REAL threshold: 50mm in world space, not normalized!
                        threshold_norm = 0.05 / max_distance  # 0.05m = 50mm in normalized space
                        accurate_i = torch.sum(vertex_dist_i < threshold_norm).item()
                    else:
                        mean_error_world = mean_error_norm  # No normalization
                        accurate_i = torch.sum(vertex_dist_i < 0.05).item()
                    
                    sample_metrics.append({
                        'pred_num': num_exists,
                        'real_prob': real_exist_prob,
                        'fake_prob': fake_exist_prob,
                        'mean_error_norm': mean_error_norm,  # Keep normalized for analysis
                        'mean_error_world': mean_error_world,  # THE REAL ERROR!
                        'accurate': accurate_i,
                        'total': V_i
                    })
                
                # Average metrics across batch - use REAL-WORLD errors!
                mean_vertex_error_norm = np.mean([m['mean_error_norm'] for m in sample_metrics])
                mean_vertex_error_world = np.mean([m['mean_error_world'] for m in sample_metrics])
                
                # Track best based on REAL error
                if mean_vertex_error_world < best_vertex_error:
                    best_vertex_error = mean_vertex_error_world
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Print with REAL-WORLD errors prominently displayed
                if max_distance:
                    print(f"Epoch {epoch+1:4d}: Loss={total_loss.item():.6f} "
                          f"(Coord={coord_loss.item():.6f}×{coord_weight:.1f}, Exist={existence_loss.item():.6f}×{existence_weight:.1f}), "
                          f"LR={current_lr:.2e}")
                    print(f"          REAL Error: {mean_vertex_error_world:.4f}m ({mean_vertex_error_world*1000:.0f}mm) "
                          f"[norm: {mean_vertex_error_norm:.4f}]")
                else:
                    print(f"Epoch {epoch+1:4d}: Loss={total_loss.item():.6f} "
                          f"(Coord={coord_loss.item():.6f}×{coord_weight:.1f}, Exist={existence_loss.item():.6f}×{existence_weight:.1f}), "
                          f"LR={current_lr:.2e}, Mean_err={mean_vertex_error_norm:.4f}")
                
                # Show SINGLE sample details (only 1 sample now)
                for idx in [0]:  # Only sample 0
                    m = sample_metrics[idx]
                    if max_distance:
                        print(f"  Sample{idx+1}: Pred_num={m['pred_num']:.0f}/{num_vertices_list[idx]}, "
                              f"RealProb={m['real_prob']:.3f}, FakeProb={m['fake_prob']:.3f}, "
                              f"Err={m['mean_error_world']:.4f}m ({m['mean_error_world']*1000:.0f}mm), "
                              f"Acc={m['accurate']}/{num_vertices_list[idx]}")
                    else:
                        print(f"  Sample{idx+1}: Pred_num={m['pred_num']:.0f}/{num_vertices_list[idx]}, "
                              f"RealProb={m['real_prob']:.3f}, FakeProb={m['fake_prob']:.3f}, "
                              f"Mean_err={m['mean_error_norm']:.4f}, Acc={m['accurate']}/{num_vertices_list[idx]}")
                
                # Progress checkpoint every 50 epochs (faster for single sample)
                if max_distance and (epoch + 1) % 50 == 0:
                    print(f"  " + "="*60)
                    if mean_vertex_error_world > 0.5:
                        print(f"  [!] WARNING: Error still > 0.5m after {epoch+1} epochs")
                        print(f"  Consider: More epochs or check data quality")
                    elif mean_vertex_error_world > 0.1:
                        print(f"  [~] Progress: Error {mean_vertex_error_world*1000:.0f}mm (target: <10mm)")
                    elif mean_vertex_error_world > 0.01:
                        print(f"  [+] Good progress! Error {mean_vertex_error_world*1000:.0f}mm (almost perfect!)")
                    else:
                        print(f"  [✓✓✓] PERFECT OVERFIT! Error {mean_vertex_error_world*1000:.0f}mm < 10mm target!")
                    print(f"  " + "="*60)
                
    # Final evaluation with detailed per-sample analysis
    print("\n" + "="*60)
    print("FINAL EVALUATION - SINGLE SAMPLE RESULT")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(points)
        pred_vertices = outputs['vertex_coords']
        existence_probs = outputs['existence_probs']
        
        for i in range(B):
            V_i = num_vertices_list[i]
            sample_id = target_samples[i]
            
            # Vertex count prediction
            exists_mask = existence_probs[i] > 0.5
            num_predicted = exists_mask.sum().item()
            
            # Existence probabilities
            real_exist_prob = existence_probs[i, :V_i].mean().item()
            fake_exist_prob = existence_probs[i, V_i:].mean().item() if V_i < model.max_vertices else 0.0
            
            # Coordinate errors
            vertex_distances = torch.norm(
                pred_vertices[i:i+1, :V_i, :] - vertices_gt_padded[i:i+1, :V_i, :], 
                dim=2
            ).squeeze()
            
            mean_error = vertex_distances.mean().item()
            median_error = vertex_distances.median().item()
            max_error = vertex_distances.max().item()
            accurate_vertices = torch.sum(vertex_distances < 0.05).item()
            
            print(f"\nSample {i+1} (ID={sample_id}):")
            print(f"  Vertex Count: Predicted={num_predicted}, Ground Truth={V_i}")
            print(f"  Existence Probs: Real={real_exist_prob:.3f}, Fake={fake_exist_prob:.3f}")
            print(f"  Coordinate Errors (m): Mean={mean_error:.4f}, Median={median_error:.4f}, Max={max_error:.4f}")
            print(f"  Accurate vertices (<5cm): {accurate_vertices}/{V_i} ({100*accurate_vertices/V_i:.1f}%)")
    
    print("\n" + "="*60)
    print(f"🎉 Training complete!")
    print(f"   Best vertex error achieved: {best_vertex_error:.6f}")
    print("="*60)
    
    # Show GPU memory
    if device.type == 'cuda':
        print_gpu_memory()
    
    return {
        'model': model,
        'best_vertex_error': best_vertex_error,
        'device_used': str(device),
        'samples_trained': 'samples 1, 10, 100, 1003, 1004, 1006, 10001, 10004, 10005, 10006, 10007, 10011',
        'batch_size': B
    }