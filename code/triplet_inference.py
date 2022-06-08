import os
import json
import time
import glob
import torch.nn as nn

import cv2
import pandas as pd
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
# from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg as LA
from tqdm import tqdm
import faiss   
from sklearn import preprocessing

from model_utils import crop_batch
from inference import Inference
from CombineModel import TripletCombineModel



          
          
if __name__=='__main__':
  model_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/'
  inference = Inference(model_path="/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/")

  classifier_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/triplet_model/'
  csv_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv'

  classifier = torch.load(classifier_path + "best_model.pth", map_location='cpu')
  classifier.eval()
  df = pd.read_csv('/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv/triplet_test.csv')
  anchor = df.anchor.to_list()
  possitive = df.positive.to_list()
  negative = df.negative.to_list()
  
  result = {
    'positive': [],
    'negative': []
  }
  for idx in tqdm(range(len(anchor))):
    with torch.no_grad():
      anchor_path = anchor[idx]
      positive_path = possitive[idx]
      negative_path = negative[idx]
      print(anchor_path, positive_path, negative_path)
      img_dict = {0: cv2.resize(cv2.imread(anchor_path), (240,320)) ,\
                  1: cv2.resize(cv2.imread(positive_path), (240,320)), \
                  2: cv2.resize(cv2.imread(negative_path), (240,320))
                  }
      result_dict = crop_batch(img_dict)
      anchor_img, positive_img, negative_img = result_dict[0][1], result_dict[1][1], result_dict[2][1]
      anchor_np, positive_np, negative_np = inference.extract_image(anchor_img), inference.extract_image(positive_img), inference.extract_image(negative_img)
      print(anchor_np.shape)
      path1, path2 = torch.tensor(anchor_np), torch.tensor(positive_np)
      print(path1.shape)
      vector1 = classifier(path1)
      vector2 = classifier(path2)
      euclid_dis = LA.norm(vector1 - vector2)
      result['positive'].append(euclid_dis)
      
      path1, path2 = torch.tensor(anchor_np), torch.tensor(negative_np)
      vector1 = classifier(path1)
      vector2 = classifier(path2)
      euclid_dis = LA.norm(vector1 - vector2)
      result['negative'].append(euclid_dis)
      # break
    res_df = pd.DataFrame.from_dict(result)
    db_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv'
    res_df.to_csv(os.path.join(db_path, 'distance.csv'),index=False)
    print(res_df.describe())
    break
  
    