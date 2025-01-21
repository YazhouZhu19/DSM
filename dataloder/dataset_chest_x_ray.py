"""
Dataset for Training and Test
Extended from ADNet code by Hansen et al.
"""
import torch
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as deftfx
import glob
import os
import SimpleITK as sitk
import random
import numpy as np
from . import image_transforms as myit
from .dataset_specifics import *
from skimage.transform import resize

import torchvision.transforms as transforms
from PIL import Image




class TestDataset(Dataset):
    def __init__(self, args):
        # Reading the paths
        if args['dataset'] == 'Chest-X-Ray':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'samples/image*'))
    
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.png')[0]))
        self.FOLD = get_folds(self.image_dirs)
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]

        
        # Split into support/query
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]  # supp_idx = 2
        self.image_dirs.pop(idx[args['supp_idx']])  # remove support
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
        
        self.label = None

    def __len__(self):
        return len(self.image_dirs)


    def __getitem__(self, idx):
        # Sample query images
        img_path = self.image_dirs[idx]
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)
        img = self.transform(img)
        
        # Load label (assuming label files have same name but different extension)
        lbl_path = img_path.split('image_')[0] + 'label_' + img_path.split('image_')[-1]
        
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)
        lbl = (lbl > 128).astype(np.uint8)  # Binary mask
        
        lbl[lbl == 200] = 1
        lbl[lbl == 500] = 2
        lbl[lbl == 600] = 3
        lbl = 1 * (lbl == self.label)
        
        sample = {'id': img_path}
        sample['image'] = img
        sample['label'] = torch.from_numpy(lbl)

        return sample

    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')

        # Load support image
        img = cv2.imread(self.support_dir, cv2.IMREAD_GRAYSCALE)
        img = Image.fromarray(img)
        img = self.transform(img)
        
        # Load support label
        lbl_path = self.support_dir.replace('images', 'labels').replace('.png', '_mask.png')
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        lbl = cv2.resize(lbl, (256, 256), interpolation=cv2.INTER_NEAREST)
        lbl = (lbl > 128).astype(np.uint8)
        
        sample = {
            'image': img,
            'label': torch.from_numpy(lbl)
        }
        
        return sample



