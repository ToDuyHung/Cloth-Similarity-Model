import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class NewDataset(Dataset):
    def __init__(self, X, y, encoder, transform=None):
        self.encoder = encoder
        self.X = X
        self.y = y
        self.articleType = self.encoder['articleType'].transform(y['articleType'])
        self.baseColour = self.encoder['baseColour'].transform(y['baseColour'])
        self.transform = transform
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        X = self.X.iloc[index]
        articleType = self.articleType[index]
        baseColour = self.baseColour[index]
        img = Image.open(X['fname'])
        
        # Backup for crop model
        img = img.resize((240,320))
        # ------
        
        # img = cv2.imread(new_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, (torch.tensor(articleType, dtype=torch.long), torch.tensor(baseColour, dtype=torch.long))
    
    def articleType_counts(self):
        return np.bincount(self.articleType)
    def baseColour_counts(self):
        return np.bincount(self.baseColour)

class DataModule(pl.LightningDataModule):
    def __init__(self, data_path, img_size, batch_size=128, sampler_func=None):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sampler_func = sampler_func
        self.sampler = None
        #  ---------------------------------- Back up-----------------------------------------
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
        self.id2articleType, self.articleType2id = {}, {}
        self.id2baseColour, self.baseColour2id = {}, {}
        self.encoder = None

    def setup(self, stage=None):
        if not self.train_ds:
            X_train = pd.read_csv(os.path.join(self.data_path, 'X_train.csv'), index_col=0)
            y_train = pd.read_csv(os.path.join(self.data_path, 'y_train.csv'), index_col=0)
            X_val = pd.read_csv(os.path.join(self.data_path, 'X_val.csv'), index_col=0)
            y_val = pd.read_csv(os.path.join(self.data_path, 'y_val.csv'), index_col=0)
                

            # print(y_train.articleType.value_counts())

            le_articleType = LabelEncoder()
            le_baseColour = LabelEncoder()

            
            self.encoder = {
            'baseColour': le_baseColour.fit(y_train['baseColour'].values),
            'articleType': le_articleType.fit(y_train['articleType'].values)
            }

            for i, class_name in enumerate(le_articleType.classes_):
                self.articleType2id[class_name] = str(i)
                self.id2articleType[str(i)] = class_name
            for i, class_name in enumerate(le_baseColour.classes_):
                self.baseColour2id[class_name] = str(i)
                self.id2baseColour[str(i)] = class_name
            # Define Dataset
            self.train_ds = NewDataset(X_train, y_train, self.encoder, transform=self.train_trans)
            self.val_ds = NewDataset(X_val, y_val, self.encoder, transform=self.val_trans)
    
            # Define Batch Sampler
            if self.sampler_func:
                self.sampler = self.sampler_func(self.train_ds)