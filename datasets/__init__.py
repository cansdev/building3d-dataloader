#!/usr/bin/python3
# _*_ coding: utf-8 _*_
# @Time    : 2023-08-11 2:41 p.m.
# @Author  : shangfeng
# @Organization: University of Calgary
# @File    : __init__.py.py
# @IDE     : PyCharm

from .building3d import Building3DReconstructionDataset


def build_dataset(dataset_config, split_set=None, logger=None):
    """
    Build Building3D dataset.
    
    Args:
        dataset_config: Dataset configuration
        split_set: Optional - specify 'train' or 'test' to return single dataset
        logger: Optional logger
        
    Returns:
        If split_set is None: dict with 'train' and 'test' datasets
        If split_set is specified: single dataset for that split
    """
    if split_set is not None:
        # Return single dataset for specified split
        return Building3DReconstructionDataset(dataset_config, split_set=split_set, logger=logger)
    else:
        # Return dict with both splits (original behavior)
        datasets_dict = {
            "train": Building3DReconstructionDataset(dataset_config, split_set="train", logger=logger),
            "test": Building3DReconstructionDataset(dataset_config, split_set="test", logger=logger),
        }
        return datasets_dict
