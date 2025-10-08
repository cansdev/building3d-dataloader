import numpy as np

from eval.ap_calculator import APCalculator
from datasets import building3d

## Example to evaluate one sample, taken from Building3D repository

# read point clouds
pc = np.loadtxt('./datasets/demo_dataset/train/xyz/10.xyz', dtype=np.float64)

# read wireframe
pd_vertices, pd_egdes = building3d.load_wireframe('./datasets/demo_dataset/train/wireframe/10.obj')
gt_vertices, gt_egdes = building3d.load_wireframe('./datasets/demo_dataset/train/wireframe/10.obj')

pd_edges_vertices = np.stack((pd_vertices[pd_egdes[:, 0]], pd_vertices[pd_egdes[:, 1]]), axis=1)
pd_edges_vertices = pd_edges_vertices[ np.arange(pd_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(pd_edges_vertices[:, :, -1]), axis=1)]

gt_edges_vertices = np.stack((gt_vertices[gt_egdes[:, 0]], gt_vertices[gt_egdes[:, 1]]), axis=1)
gt_edges_vertices = gt_edges_vertices[ np.arange(gt_edges_vertices.shape[0])[:, np.newaxis], np.flip(np.argsort(pd_edges_vertices[:, :, -1]), axis=1)]

ap_calculator = APCalculator(distance_thresh=1)

batch = dict()
batch['predicted_vertices'] = pd_vertices[np.newaxis,:]
batch['predicted_edges'] = pd_egdes[np.newaxis,:]
batch['pred_edges_vertices'] = pd_edges_vertices.reshape((1, -1, 2, 3))

batch['wf_vertices'] = gt_vertices[np.newaxis,:]
batch['wf_edges'] = gt_egdes[np.newaxis,:]
batch['wf_edges_vertices'] = gt_edges_vertices.reshape((1, -1, 2, 3))

ap_calculator.compute_metrics(batch)
ap_calculator.output_accuracy()

