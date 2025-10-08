#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# Training with DGCNN model and CUDA support

import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.cuda_utils import to_cuda, print_gpu_memory
from models.dgcnn_model import BasicDGCNN
from losses.vertex_loss_mse import VertexLossMSE
from losses.vertex_loss_hungarian import VertexLossHungarian

def main_training_setup():
    """Setup training configuration"""
    print("Training setup initialized with DGCNN")
    return True

def train_on_real_dataset(train_loader, device, train_config):
    # Initialize DGCNN model from config
    model = BasicDGCNN(
        input_dim=train_config['model']['input_dim'],
        k=train_config['model']['k_neighbors'],
        max_vertices=train_config['model']['max_vertices']
    )
    model = model.to(device)
    
    # Initialize optimizer from config
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['optimizer']['lr'],
        weight_decay=train_config['optimizer']['weight_decay']
    )
    
    # Initialize scheduler from config
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_config['scheduler']['T_max'],
        eta_min=train_config['scheduler']['eta_min']
    )
    
    # Initialize loss function from config
    use_hungarian = train_config['loss']['use_hungarian']
    if use_hungarian:
        print("Using Hungarian matching loss (DETR-style)")
        criterion = VertexLossHungarian(
            coord_weight=train_config['loss']['coord_weight'],
            existence_weight=train_config['loss']['existence_weight']
        )
    else:
        print("Using MSE loss")
        criterion = VertexLossMSE(
            coord_weight=train_config['loss']['coord_weight'],
            existence_weight=train_config['loss']['existence_weight']
        )
    criterion = criterion.to(device)
    
    print(f"DGCNN model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    target_samples = [1, 10, 100, 1003] 
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
        print(f"Could not find all 4 samples! Found: {list(collected_samples.keys())}")
        return {'error': f'Only found {len(collected_samples)} samples: {list(collected_samples.keys())}'}
    
    # Now build the training batch from collected samples
    print(f"Found all 4 samples: {list(collected_samples.keys())}")
    
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
    vertices_gt = training_batch['wf_vertices']  # [4, V_padded, 3] ground truth vertex coordinates (may contain -10 padding)
    
    print(f"Training on FOUR SAMPLES: {list(collected_samples.keys())}")
    print(f"   Point clouds: {points.shape}")
    print(f"   GT vertices (with padding): {vertices_gt.shape}")
    
    # Filter out padded vertices (padded with -10 by collate_batch)
    B = points.shape[0]  # Batch size = 4
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
            print(f"   Sample {i+1}: Filtered padding: {V_i} REAL vertices (removed {vertices_gt.shape[1] - V_i} padded)")
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
    
    model.train()
    num_epochs = train_config['num_epochs']
    
    best_loss = float('inf')
    best_vertex_error = float('inf')
    
    for epoch in range(num_epochs):
        # Set epoch for deterministic-but-varying random sampling
        # This ensures different point subsets each epoch while maintaining reproducibility
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        optimizer.zero_grad()
        
        # Forward pass - now predicts absolute coordinates directly
        outputs = model(points)
        pred_coords = outputs['vertex_coords']          # [B, max_vertices, 3] - absolute coordinates
        existence_logits = outputs['existence_logits']  # [B, max_vertices] - raw logits
        existence_probs = outputs['existence_probs']    # [B, max_vertices] - sigmoid probabilities
        pred_num_vertices = outputs['num_vertices']     # [B] - dynamic count from threshold
        
        # Create per-vertex existence targets (binary mask)
        # For each sample, first N vertices should exist (1.0), rest should not (0.0)
        existence_target = torch.zeros(B, model.max_vertices).to(device)
        for i in range(B):
            V_i = num_vertices_list[i]
            existence_target[i, :V_i] = 1.0  # First V_i vertices exist
        
        # Prepare predictions and targets for loss computation
        predictions = {
            'vertex_offsets': pred_coords,  # Using absolute coords directly
            'existence_logits': existence_logits,
            'pc_centroid': torch.zeros(B, 1, 3, device=device)  # Dummy
        }
        
        targets = {
            'vertices_gt_offsets': vertices_gt_padded,  # Using absolute coords directly
            'existence_target': existence_target
        }
        
        # Compute loss using the loss module
        loss_dict = criterion(predictions, targets, num_vertices_list)
        
        total_loss = loss_dict['total_loss']
        coord_loss = loss_dict['coord_loss']
        existence_loss = loss_dict['existence_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping from config
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config['gradient_clip_norm'])
        
        optimizer.step()
        scheduler.step()
        
        # Track best model
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        if (epoch + 1) % 5 == 0 or epoch < 20:
            with torch.no_grad():
                # Use predicted absolute coordinates directly
                pred_vertices_abs = pred_coords  # [B, max_vertices, 3]
                
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
                    
                    vertex_dist_i = torch.norm(pred_vertices_abs[i:i+1, :V_i, :] - vertices_gt_padded[i:i+1, :V_i, :], dim=2)
                    mean_error_world = vertex_dist_i.mean().item()
                    
                    # Model already predicts in absolute world coordinates (not normalized)
                    # So the error is already in meters - no scaling needed!
                    accurate_i = torch.sum(vertex_dist_i < 0.05).item()
                    
                    sample_metrics.append({
                        'pred_num': num_exists,
                        'real_prob': real_exist_prob,
                        'fake_prob': fake_exist_prob,
                        'mean_error_world': mean_error_world,  # Error in meters (already in world coords)
                        'accurate': accurate_i,
                        'total': V_i
                    })
                
                # Average metrics across batch
                mean_vertex_error_world = np.mean([m['mean_error_world'] for m in sample_metrics])
                
                # Track best based on REAL error
                if mean_vertex_error_world < best_vertex_error:
                    best_vertex_error = mean_vertex_error_world
                
                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Print training progress
                print(f"Epoch {epoch+1:4d}: Loss={total_loss.item():.6f} "
                      f"(Coord={coord_loss.item():.6f}×{criterion.coord_weight:.1f}, Exist={existence_loss.item():.6f}×{criterion.existence_weight:.1f}), "
                      f"LR={current_lr:.2e}")
                
    # Final evaluation
    print("\n" + "="*60)
    
    model.eval()
    with torch.no_grad():
        outputs = model(points, return_absolute=True)  # Get absolute coords for final eval
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
    print(f"Training complete!")
    print(f"   Best vertex error achieved: {best_vertex_error:.6f}")
    print("="*60)
    
    # Show GPU memory
    if device.type == 'cuda':
        print_gpu_memory()
    
    return {
        'model': model,
        'best_vertex_error': best_vertex_error,
        'device_used': str(device),
        'samples_trained': 'samples 1, 10, 100, 1003',
        'batch_size': B
    }