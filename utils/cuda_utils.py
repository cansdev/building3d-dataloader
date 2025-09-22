#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# CUDA utility for device detection

import torch

def get_device():
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: Device to use for computation
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
        return device
    else:
        device = torch.device('cpu')
        print(f"CUDA not available, using CPU")
        return device

def to_cuda(data, device):
    """
    Move data to specified device.
    
    Args:
        data: Data to move (tensor, dict, list, etc.)
        device: Target device
        
    Returns:
        Data moved to device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: to_cuda(value, device) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_cuda(item, device) for item in data)
    else:
        return data

def print_gpu_memory():
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory:{reserved:.2f}GB reserved")

def get_detailed_gpu_info():
    """Get detailed GPU information."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        return {
            'name': device_props.name,
            'total_memory_gb': device_props.total_memory / 1024**3,
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'free_gb': (device_props.total_memory - torch.cuda.memory_reserved()) / 1024**3
        }
    return None