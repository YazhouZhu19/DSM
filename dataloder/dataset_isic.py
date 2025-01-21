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
        # 读取图像路径

        if args['dataset'] == 'ISIC':
            self.image_dirs = glob.glob(os.path.join(args['data_dir'], 'samples/image*'))

        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.jpg')[0]))
        
        # 获取测试集的fold
        self.FOLD = self.get_folds()
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx in self.FOLD[args['eval_fold']]]
        
        
        # 设置目标尺寸
        self.target_size = (256, 256)  # 可以根据需要调整
        
        # 从support_idx选择支持集图像
        idx = np.arange(len(self.image_dirs))
        self.support_dir = self.image_dirs[idx[args['supp_idx']]]
        self.image_dirs.pop(idx[args['supp_idx']])
        self.label = None

    def __len__(self):
        return len(self.image_dirs)

    def get_folds(self):
        # 划分数据集fold，可以根据需要修改划分方式
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
        # 加载查询图像
        img_path = self.image_dirs[idx]
        img = Image.open(img_path).convert('L')  # 转换为灰度图
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img = np.array(img)

        # 标准化
        img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.stack(3 * [img], axis=0)  # 转换为3通道

        # 加载对应的标签
        # lbl_path = img_path.replace('image_', 'label_')
        lbl_path = img_path.replace('image_', 'label_').replace('.jpg', '.png')
        lbl = Image.open(lbl_path).convert('L')
        lbl = lbl.resize(self.target_size, Image.Resampling.NEAREST)
        lbl = np.array(lbl)
        
        # 二值化标签
        lbl = (lbl > 0).astype(np.float32)
        
        sample = {'id': img_path}
        
        # 只选择包含标签的切片
        idx = lbl.sum((0, 1)) > 0
        sample['image'] = torch.from_numpy(img).float()
        sample['label'] = torch.from_numpy(lbl).float()


        return sample


    def getSupport(self, label=None, all_slices=True, N=None):
        if label is None:
            raise ValueError('Need to specify label class!')
            
        # 加载支持集图像
        img = Image.open(self.support_dir).convert('L')
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img = np.array(img)
        
        # 标准化
        img = (img - img.mean()) / (img.std() + 1e-8) 
        img = np.stack(3 * [img], axis=0)
        

        # 加载支持集标签
        # lbl_path = self.support_dir.replace('image_', 'label_')
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




class TrainDataset(Dataset):
    def __init__(self, args):
        self.n_shot = args['n_shot']
        self.n_way = args['n_way']
        self.n_query = args['n_query']
        self.max_iter = args['max_iter']
        self.min_size = args.get('min_size', 100)
        
        # 设置目标尺寸
        self.target_size = (256, 256)  # 可以根据需要调整
        
        # 读取图像和标签路径
        self.image_dirs = glob.glob(os.path.join('samples', 'image_*.png'))
        self.label_dirs = glob.glob(os.path.join('samples', 'label_*.png'))
        
        self.image_dirs = sorted(self.image_dirs, key=lambda x: int(x.split('_')[-1].split('.png')[0]))
        self.label_dirs = sorted(self.label_dirs, key=lambda x: int(x.split('_')[-1].split('.png')[0]))

        # 移除测试fold
        self.FOLD = self.get_folds()
        self.image_dirs = [elem for idx, elem in enumerate(self.image_dirs) if idx not in self.FOLD[args['eval_fold']]]
        self.label_dirs = [elem for idx, elem in enumerate(self.label_dirs) if idx not in self.FOLD[args['eval_fold']]]

    def get_folds(self):
        # 划分数据集fold，与TestDataset相同
        n_samples = 704
        indices = np.arange(n_samples)
        folds = {
            0: indices[0:141],
            1: indices[141:282],
            2: indices[282:423],
            3: indices[423:564],
            4: indices[564:704]
        }
        return folds

    def __len__(self):
        return self.max_iter

    def apply_transforms(self, image, label):
        # 数据增强
        if random.random() > 0.5:
            # 随机水平翻转
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        if random.random() > 0.5:
            # 随机垂直翻转
            image = TF.vflip(image)
            label = TF.vflip(label)
            
        # 随机旋转
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)
            
        return image, label

    def __getitem__(self, idx):
        # 随机选择病人索引
        pat_idx = random.choice(range(len(self.image_dirs)))
        
        # 加载图像
        img = Image.open(self.image_dirs[pat_idx]).convert('L')
        img = img.resize(self.target_size, Image.Resampling.BILINEAR)
        
        # 加载标签
        lbl = Image.open(self.label_dirs[pat_idx]).convert('L')
        lbl = lbl.resize(self.target_size, Image.Resampling.NEAREST)
        
        # 转换为numpy数组
        img = np.array(img)
        lbl = np.array(lbl)
        
        # 标准化图像
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # 二值化标签
        lbl = (lbl > 0).astype(np.float32)
        
        # 准备support和query样本
        sup_img = np.stack(3 * [img], axis=0)[None,]  # [1, 3, H, W]
        sup_lbl = lbl[None, None,]  # [1, 1, H, W]
        
        qry_img = np.stack(3 * [img], axis=0)[None,]  # [1, 3, H, W]
        qry_lbl = lbl[None,]  # [1, H, W]
        
        # 转换为tensor
        sup_img = torch.from_numpy(sup_img).float()
        sup_lbl = torch.from_numpy(sup_lbl).float()
        qry_img = torch.from_numpy(qry_img).float()
        qry_lbl = torch.from_numpy(qry_lbl).float()
        
        # 应用数据增强
        if random.random() > 0.5:
            sup_img = sup_img * (1 + 0.1 * torch.randn_like(sup_img))  # 添加随机噪声
        else:
            qry_img = qry_img * (1 + 0.1 * torch.randn_like(qry_img))
        
        sample = {
            'support_images': sup_img,
            'support_fg_labels': sup_lbl,
            'query_images': qry_img,
            'query_labels': qry_lbl,
        }
        
        return sample



