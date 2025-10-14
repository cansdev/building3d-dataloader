#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# Training with DGCNN model and CUDA support

import torch
import torch.nn as nn
import numpy as np
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.cuda_utils import to_cuda, print_gpu_memory
from models.dgcnn_model import BasicDGCNN
from losses.vertex_loss_mse import VertexLossMSE
from losses.vertex_loss_hungarian import VertexLossHungarian

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
    
    # Try to resume from checkpoint
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    start_epoch = 0
    best_loss = float('inf')
    best_vertex_error = float('inf')
    
    if os.path.exists(checkpoint_path):
        print(f"\nFound checkpoint at {checkpoint_path}, resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        best_vertex_error = checkpoint.get('best_vertex_error', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best loss so far: {best_loss:.6f}")
        print(f"Best vertex error so far: {best_vertex_error:.4f}m")
    else:
        print("\nNo checkpoint found. Starting training from scratch.")
    
    print("\nPreprocessing details shown only for epoch 0...")
    print("=" * 60)
    
    model.train()
    num_epochs = train_config['num_epochs']
    max_vertices = model.max_vertices
    
    for epoch in range(start_epoch, num_epochs):
        # Set epoch for deterministic-but-varying random sampling
        if hasattr(train_loader.dataset, 'set_epoch'):
            train_loader.dataset.set_epoch(epoch)
        
        epoch_losses = []
        epoch_coord_losses = []
        epoch_exist_losses = []
        epoch_vertex_errors = []
        
        # Train on all batches
        for batch_idx, batch_data in enumerate(train_loader):
            # Move batch to device
            batch_data = to_cuda(batch_data, device)
            
            # Extract data
            points = batch_data['point_clouds']  # [B, N, 5]
            vertices_gt = batch_data['wf_vertices']  # [B, V_padded, 3]
            B = points.shape[0]
            
            # Filter out padded vertices
            valid_vertices_list = []
            num_vertices_list = []
            
            for i in range(B):
                valid_mask_i = (vertices_gt[i, :, 0] > -9.0)
                valid_vertices_i = vertices_gt[i, valid_mask_i, :]
                V_i = valid_vertices_i.shape[0]
                valid_vertices_list.append(valid_vertices_i)
                num_vertices_list.append(V_i)
            
            # Prepare ground truth - properly padded
            vertices_gt_padded = torch.zeros(B, max_vertices, 3, device=device)
            for i in range(B):
                V_i = num_vertices_list[i]
                if V_i > max_vertices:
                    V_i = max_vertices
                    num_vertices_list[i] = max_vertices
                vertices_gt_padded[i, :V_i, :] = valid_vertices_list[i][:V_i, :]
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(points)
            pred_coords = outputs['vertex_coords']
            existence_logits = outputs['existence_logits']
            existence_probs = outputs['existence_probs']
            
            # Create existence targets
            existence_target = torch.zeros(B, max_vertices).to(device)
            for i in range(B):
                V_i = num_vertices_list[i]
                existence_target[i, :V_i] = 1.0
            
            # Prepare predictions and targets for loss
            predictions = {
                'vertex_offsets': pred_coords,
                'existence_logits': existence_logits,
                'pc_centroid': torch.zeros(B, 1, 3, device=device)
            }
            
            targets = {
                'vertices_gt_offsets': vertices_gt_padded,
                'existence_target': existence_target
            }
            
            # Compute loss
            loss_dict = criterion(predictions, targets, num_vertices_list)
            total_loss = loss_dict['total_loss']
            coord_loss = loss_dict['coord_loss']
            existence_loss = loss_dict['existence_loss']
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config['gradient_clip_norm'])
            optimizer.step()
            
            # Track metrics
            epoch_losses.append(total_loss.item())
            epoch_coord_losses.append(coord_loss.item())
            epoch_exist_losses.append(existence_loss.item())
            
            # Compute vertex error for this batch
            with torch.no_grad():
                batch_vertex_errors = []
                for i in range(B):
                    V_i = num_vertices_list[i]
                    vertex_dist_i = torch.norm(pred_coords[i:i+1, :V_i, :] - vertices_gt_padded[i:i+1, :V_i, :], dim=2)
                    batch_vertex_errors.append(vertex_dist_i.mean().item())
                epoch_vertex_errors.extend(batch_vertex_errors)
        
        # Step scheduler once per epoch
        scheduler.step()
        
        # Compute epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_coord_loss = np.mean(epoch_coord_losses)
        avg_exist_loss = np.mean(epoch_exist_losses)
        avg_vertex_error = np.mean(epoch_vertex_errors)
        
        # Track best metrics
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if avg_vertex_error < best_vertex_error:
            best_vertex_error = avg_vertex_error
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_vertex_error': best_vertex_error,
                'train_config': train_config
            }, checkpoint_save_path)
            
            # Also update the latest checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'best_vertex_error': best_vertex_error,
                'train_config': train_config
            }, checkpoint_path)
            
            print(f"  → Checkpoint saved at epoch {epoch + 1}")
        
        # Print progress every epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:4d}/{num_epochs}: Loss={avg_loss:.6f} "
              f"(Coord={avg_coord_loss:.6f}, Exist={avg_exist_loss:.6f}), "
              f"VertexError={avg_vertex_error:.4f}m, LR={current_lr:.2e}")
    
    print("\n" + "="*60)
    print(f"Training complete!")
    print(f"   Best loss: {best_loss:.6f}")
    print(f"   Best vertex error: {best_vertex_error:.4f}m")
    print("="*60)
    
    # Show GPU memory
    if device.type == 'cuda':
        print_gpu_memory()
    
    return {
        'model': model,
        'best_loss': best_loss,
        'best_vertex_error': best_vertex_error,
        'device_used': str(device),
        'num_epochs': num_epochs
    }