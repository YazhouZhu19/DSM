import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


class TestDataset(Dataset):
    def __init__(self, args):

        if args['dataset'] == 'ISIC':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'samples/image*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
        
        self.FOLD = self.get_folds()
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]
        
        
        self.target_size = (256, 256) 
        
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]
        self.image_dirs.pop(idx[args['supp_idx']])
        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def get_folds(self):
        n_samples = 2594 
        indices = np.arange(n_samples)
        folds = {
            0: indices[0:5000],  
            1: indices[1960:11132],  
            2: indices[11132:13189],  
            3: indices[13187:14609],  
            4: indices[14605:16072]  
        }
        return folds

    def __getitem__(self, idx):
        img_path = self.image_dirs[idx]
        img = Image.open(img_path).convert('L')  
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img = np.array(img)

        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.stack(3 * [img], axis=0) 

        lbl_path = img_path.replace('image_', 'label_').replace('.jpg', '.png')
        lbl = Image.open(lbl_path).convert('L')
        lbl = lbl.resize(self.target_size, Image.Resampling.NEAREST)
        lbl = np.array(lbl)
        
        lbl = (lbl > 0).astype(np.float32)
        
        sample = {'id': img_path}
        
        idx = lbl.sum((0, 1)) > 0
        sample['image'] = torch.from_numpy(img).float()
        sample['label'] = torch.from_numpy(lbl).float()
        
        return sample


    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')
            
        img = Image.open(self.support_dir).convert('L')
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img = np.array(img)
        
        img = (img - img.mean()) / (img.std() + 1e-8) 
        img = np.stack(3 * [img], axis=0)
        
        lbl_path = self.support_dir.replace('image_', 'label_').replace('.jpg', '.png')
        lbl = Image.open(lbl_path).convert('L')
        lbl = lbl.resize(self.target_size, Image.Resampling.NEAREST)
        lbl = np.array(lbl)
        lbl = (lbl > 0).astype(np.float32)

        sample = {}
        if all_slices:
            sample['image'] = torch.from_numpy(img).float()
            sample['label'] = torch.from_numpy(lbl).float()
        else:
            if N is None:
                raise ValueError('Need to specify number of labeled slices!')
            sample['image'] = torch.from_numpy(img).float()
            sample['label'] = torch.from_numpy(lbl).float()
    

        return sample






