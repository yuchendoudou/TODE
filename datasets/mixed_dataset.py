import os
import os.path as osp
from glob import glob
import numpy as np
import cv2
import h5py

import torch
from torch.utils.data import Dataset


# My libraries
from .cleargrasp import ClearGraspSynthetic
from .omniverse_object import OmniverseObject



class MixedDataset(Dataset):
    def __init__(self, cleargrasp_root_dir, omniverse_root_dir, split = 'train', **kwargs):

        self.cleargrasp_syn_dataset = ClearGraspSynthetic(cleargrasp_root_dir, split, **kwargs)
        self.omniverse_dataset = OmniverseObject(omniverse_root_dir, split, **kwargs)
        self.cleargrasp_syn_len = self.cleargrasp_syn_dataset.__len__()
        self.omniverse_len = self.omniverse_dataset.__len__()

    def __getitem__(self, idx):
        if idx < self.cleargrasp_syn_len:
            return self.cleargrasp_syn_dataset.__getitem__(idx)
        else:
            return self.omniverse_dataset.__getitem__(idx-self.cleargrasp_syn_len)


    def __len__(self):
        return self.cleargrasp_syn_len + self.omniverse_len
