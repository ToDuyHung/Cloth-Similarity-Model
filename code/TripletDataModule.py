import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
# from inference import Inference
import cv2
# from model_utils import crop_batch

# model_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/'
# inference = Inference(model_path="/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/")

class NewDataset(Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X.iloc[index]
        # anchor_path = X['anchor']
        # positive_path = X['positive']
        # negative_path = X['negative']
        # img_dict = {0: cv2.resize(cv2.imread(anchor_path), (240,320)) ,\
        #             1: cv2.resize(cv2.imread(positive_path), (240,320)), \
        #             2: cv2.resize(cv2.imread(negative_path), (240,320))
        #             }
        # result_dict = crop_batch(img_dict)
        # anchor_img, positive_img, negative_img = result_dict[0][1], result_dict[1][1], result_dict[2][1]
        # return (inference.extract_image(anchor_img), inference.extract_image(positive_img), inference.extract_image(negative_img))
        return (X['anchor'], X['positive'], X['negative'])

class TripletDataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_size, batch_size=128, sampler_func=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sampler_func = sampler_func
        self.sampler = None
        self.train_trans = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(.8, 1.2)),
            transforms.Resize(img_size + 20),
            transforms.RandomCrop(img_size),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.05),
            transforms.GaussianBlur(kernel_size=(5, 5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.val_trans = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.train_ds, self.val_ds = None, None

    def setup(self, stage=None):
        if not self.train_ds:
            # X_train = pd.read_csv(os.path.join(self.data_path, 'triplet_train.csv'), index_col=0)
            # X_val = pd.read_csv(os.path.join(self.data_path, 'triplet_val.csv'), index_col=0)
            
            X_train = pd.read_hdf(os.path.join(self.data_path, 'train.h5'), key='train')
            X_val = pd.read_hdf(os.path.join(self.data_path, 'val.h5'), key='val')

            # Define Dataset
            self.train_ds = NewDataset(X_train, transform=self.train_trans)
            self.val_ds = NewDataset(X_val, transform=self.val_trans)
    
            # Define Batch Sampler
            if self.sampler_func:
                self.sampler = self.sampler_func(self.train_ds)