import pandas as pd
from glob import glob
import os
from sklearn.model_selection import train_test_split
import random
import itertools
from functools import reduce
from inference import Inference
import cv2
from model_utils import crop_batch
from tqdm import tqdm

# model_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/'
# inference = Inference(model_path="/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/")

# df_dict = {
#     'anchor': [],
#     'positive': [],
#     'negative': []
# }

# anchor_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/last_val_data/anchor'
# negative_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/last_val_data/negative'
# positive_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/last_val_data/positive'
csv_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv'

# for anchor in tqdm(glob(os.path.join(anchor_path, '*.jpg'))):
#     anchor_name = anchor.split('/')[-1]
#     anchor_name = anchor_name.split('.')[0]
#     # print(anchor_name)
    
    
#     list1 = [anchor] + glob(os.path.join(positive_path, f'{anchor_name} (*).jpg'))
#     list2 = glob(os.path.join(negative_path, f'{anchor_name} (*).jpg'))
    
#     list1 = list(itertools.combinations(list1, 2))
#     list2 = reduce(lambda x,y: x + [y] * 3, list2, [])
#     # print(len(list1), len(list2))
    
#     for idx in range(len(list1)):
#         anchor_path = list1[idx][0]
#         positive_path = list1[idx][1]
#         negative_path = list2[0]
#         img_dict = {0: cv2.resize(cv2.imread(anchor_path), (240,320)) ,\
#                     1: cv2.resize(cv2.imread(positive_path), (240,320)), \
#                     2: cv2.resize(cv2.imread(negative_path), (240,320))
#                     }
#         result_dict = crop_batch(img_dict)
#         anchor_img, positive_img, negative_img = result_dict[0][1], result_dict[1][1], result_dict[2][1]
#         anchor_np, positive_np, negative_np = inference.extract_image(anchor_img), inference.extract_image(positive_img), inference.extract_image(negative_img)
#         df_dict['anchor'].append(anchor_np)
#         df_dict['positive'].append(positive_np)
#         df_dict['negative'].append(negative_np)

# df = pd.DataFrame.from_dict(df_dict)
# # # print(df)

# train, test = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
# train, val = train_test_split(train, test_size=0.2, shuffle=True, random_state=42)

# # print(len(train),len(val), len(test))


# # train.to_csv(new_folder + 'triplet_train.csv')
# # test.to_csv(new_folder + 'triplet_test.csv')
# # val.to_csv(new_folder + 'triplet_val.csv')

# train.to_hdf(os.path.join(csv_path, 'train.h5'), key='train', mode='w')  
# val.to_hdf(os.path.join(csv_path, 'val.h5'), key='val', mode='w')  
# test.to_hdf(os.path.join(csv_path, 'test.h5'), key='test', mode='w')  


# Test ---------------------
df_train = pd.read_hdf(os.path.join(csv_path, 'train.h5'), key='train')
print(type(df_train.iloc[0,0]))
print(df_train.iloc[0,0].shape)