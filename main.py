import os

from datasets import build_dataset
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader

def cfg_from_yaml_file(cfg_file):
        with open(cfg_file, 'r') as f:
            try:
                new_config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                new_config = yaml.load(f)

        cfg = EasyDict(new_config)
        return cfg

def main():

    dataset_config = cfg_from_yaml_file('datasets/dataset_config.yaml')
    #print(dataset_config)

    building3D_dataset = build_dataset(dataset_config.Building3D)
    #print(dir(building3D_dataset['train']))
    #print(building3D_dataset["test"].wireframe_files)

    train_loader = DataLoader(building3D_dataset['train'], batch_size=3, shuffle=True, drop_last=True, collate_fn=building3D_dataset['train'].collate_batch)
    # debugging batches
    if False:
        for batch in train_loader:
            print(batch)
            print('point_clouds ', batch['point_clouds'].shape)
            print('wf_vertices', batch['wf_vertices'].shape)
            print('wf_edges', batch['wf_edges'].shape)
            print('centroid', batch['centroid'].shape)
            print('max_distance', batch['max_distance'].shape)
            print('scan_idx', batch['scan_idx'].shape)
            print(batch['wf_edges'])
            print(batch['centroid'])
            print(batch['wf_vertices'])
            break
    

if __name__ == "__main__":
    main()