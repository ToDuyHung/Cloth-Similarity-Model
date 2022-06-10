import cv2
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
# ------------------------------------------ Create crop batch dataset --------------------------------------
theme_ls = glob.glob('theme/*.jpg')
# print(np.random.choice(theme_ls))
def read_and_add_noise(img):
    h, w, c = img.shape
    img = img[:,int(0.2*w):int(0.8*w)]
    h, w, c = img.shape
    background_path = np.random.choice(theme_ls)
    background=cv2.resize(cv2.imread(background_path), (w, h))
    # print(background)
    a = (img[:,:,0]==255) & (img[:,:,1]==255) & (img[:,:,2]==255)
    a = np.expand_dims(a, axis=2)
    a = np.repeat(a, 3, axis=2)
    img = np.where(a, background, img)
    return img

crop_path = 'crop_images'
X_train = pd.read_csv('datacsv/X_train.csv')
X_val = pd.read_csv('datacsv/X_val.csv')
X_test = pd.read_csv('datacsv/X_test.csv')

for X in [X_train, X_val, X_test]:
    for fname in tqdm(X.fname):
        img = cv2.resize(cv2.imread(fname), (240,320))
        # Center crop
        # h, w, c = img.shape
        # img = img[:,int(0.2*w):int(0.8*w)]
        img = read_and_add_noise(img)
        cv2.imwrite(os.path.join(crop_path, fname.split('/')[-1]), img)
        # break
    
# ------------------------------------------ Inference -----------------------------------------------------
# img_dict = {'a.jpg': cv2.resize(cv2.imread('b.png'), (240,320))}
# result_dict = crop_batch(img_dict)
# cv2.imwrite('a.jpg', result_dict['a.jpg'][1])