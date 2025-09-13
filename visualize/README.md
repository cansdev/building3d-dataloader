# Visualization Scripts

This folder contains visualization scripts for the Building3D project.

## Scripts

### `visualize.py`
Original visualization script for the Building3D dataset. Shows point clouds and wireframe data.

**Usage:**
```bash
python visualize/visualize.py
```

### `visualize_pointnet2.py`
PointNet2 corner detection visualization script. Shows:
- Original point cloud
- Ground truth corners (wireframe vertices)
- PointNet2 predicted corners
- Comparison between ground truth and predictions

**Usage:**

**Interactive mode:**
```bash
python visualize/visualize_pointnet2.py
```

**Non-interactive mode (specific sample):**
```bash
python visualize/visualize_pointnet2.py --sample 0 --split test --threshold 0.5
```

**Arguments:**
- `--sample`: Specific sample index to visualize (if not provided, interactive mode)
- `--threshold`: Threshold for corner predictions (default: 0.5)
- `--split`: Dataset split to use - 'train' or 'test' (default: test)

## Requirements

- Trained PointNet2 model must exist at `../output/corner_detection_model.pth`
- matplotlib for 3D visualization
- All dependencies from the main project

## Integration

The `visualize_pointnet2.py` script is automatically called after training when you run `main.py` and choose to run visualization.
