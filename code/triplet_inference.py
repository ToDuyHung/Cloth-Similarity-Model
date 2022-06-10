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


class TripletCombineInference:
    
  def __init__(self, model_path: str, classifier_path: str, threshold = 4.1, device=None):
    """
    🤗 Constructor for the image classifier trainer of TorchVision
    """
    self.img_size = 224
    self.resize_shape = (240,320)
    self.threshold = threshold
    self.db_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset'
    self.use_gpu = False
    if device == None:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device
    self.model_path = model_path if model_path.endswith("/") else model_path + "/"
    self.inference = Inference(model_path=self.model_path)
    
    self.classifier_path = classifier_path if classifier_path.endswith("/") else classifier_path + "/"
    self.classifier = torch.load(self.classifier_path + "best_model.pth", map_location=self.device)
    self.classifier.eval()
    
  def extract_image(self, img_path: str, save_preview=False):
    """
    🤔 Predict from one image at the numpy array format or string format
    """
    img_dict = {0: cv2.resize(cv2.imread(img_path), (240,320))}
    result_dict = crop_batch(img_dict)
    img = result_dict[0][1]
    img_np = self.inference.extract_image(img)
    img_np = torch.tensor(img_np)
    with torch.no_grad():
        return self.classifier(img_np)

  def compare(self, img_path1: str, img_path2: str):
    '''
    Return True if img1 and img2 are similar, otherwise False
    '''
    vector1 = self.extract_image(img_path1)
    vector2 = self.extract_image(img_path2)
    euclid_dis = LA.norm(vector1 - vector2)
    # print(f'Cosine similarity: {cos_sim}')
    if euclid_dis < self.threshold:
      return True
    else:
      return False
    
  def gen_db(self, dataset_path:str):
    start = time.time()
    embedding = {}
    for dir in tqdm(os.listdir(dataset_path)):
        embedding[dir] = []
        for img_file in tqdm(glob.glob(os.path.join(dataset_path, dir, '*.jpg'))):
            print(img_file)
            vector = self.extract_image(img_file)
            vector = [list(vector[0].astype(dtype=np.float64))]
            # print(type(vector), type(vector[0]))
            embedding[dir].append(vector)
    with open(os.path.join(self.db_path, 'db.json'), 'w', encoding='utf-8') as f:
        json.dump(embedding, f)
    print('Time embedding: ', time.time() - start)
    
  def load_db(self):
      with open(os.path.join(self.db_path, 'db.json'), 'r') as f:
          db = json.load(f)
      first_time = True
      list_feature = []
      list_id = []
      list_len = []
      for k,v in db.items():
          list_id.append(k)
          list_len.append(len(v))
          if first_time:
              d = np.array(v[0]).shape[1]
              print(np.array(v[0]).shape)
              
              # If use IndexIVFFlat:
              # nlist = 100 if np.array(v[0]).shape[0] > 100 else 2
              # quantizer = faiss.IndexFlatIP(d)
              # index = faiss.IndexIVFFlat(quantizer, d, nlist)
              # 
              # Elif use IndexFlatIP:
              index = faiss.IndexFlatIP(d)
              # 
              if self.use_gpu:
                  device =  faiss.StandardGpuResources()  # use a single GPU
                  index = faiss.index_cpu_to_gpu(device, 0, index)
              first_time = False
          for feature in v:
              list_feature.append(np.array(feature).astype('float32').reshape(1,1024))

      list_feature = np.concatenate(list_feature , axis=0)
      list_feature_new = preprocessing.normalize(list_feature, norm='l2')
      index.add(list_feature_new)
      return list_len, list_id, index

  def identification(self, img_file, list_len, list_id, index):
      res = None
      vectors = self.extract_image(img_file)
      max = 0
      for vector in vectors:
          xq = np.array(vector).astype('float32').reshape(1,1024)
          xq = preprocessing.normalize(xq, norm='l2')
          start_search = time.time()
          distances, indices = index.search(xq, 1)
          print('End search: ',time.time()-start_search)
          position = indices[0][0]
          sum = 0
          for idx in range(len(list_id)):
              sum += list_len[idx]
              if position < sum:
                  PID = list_id[idx]
                  break
          print(f'{PID}: {distances[0][0]}')
          # print(distances)
          if distances[0][0] < self.threshold or distances[0][0] < max:
              continue
          max = distances[0][0]
          res = PID
      return res
          
          
if __name__=='__main__':
  model_path = 'crop_pytorch_model/'
  # inference = Inference(model_path="crop_pytorch_model/")

  classifier_path = 'triplet_model/'
  # csv_path = 'datacsv'

  # classifier = torch.load(classifier_path + "best_model.pth", map_location='cpu')
  # classifier.eval()
  # df = pd.read_csv('datacsv/triplet_test.csv')
  # anchor = df.anchor.to_list()
  # possitive = df.positive.to_list()
  # negative = df.negative.to_list()
  
  # result = {
  #   'positive': [],
  #   'negative': []
  # }
  # for idx in tqdm(range(len(anchor))):
  #   with torch.no_grad():
  #     anchor_path = anchor[idx]
  #     positive_path = possitive[idx]
  #     negative_path = negative[idx]
  #     print(anchor_path, positive_path, negative_path)
  #     img_dict = {0: cv2.resize(cv2.imread(anchor_path), (240,320)) ,\
  #                 1: cv2.resize(cv2.imread(positive_path), (240,320)), \
  #                 2: cv2.resize(cv2.imread(negative_path), (240,320))
  #                 }
  #     result_dict = crop_batch(img_dict)
  #     anchor_img, positive_img, negative_img = result_dict[0][1], result_dict[1][1], result_dict[2][1]
  #     anchor_np, positive_np, negative_np = inference.extract_image(anchor_img), inference.extract_image(positive_img), inference.extract_image(negative_img)
  #     print(anchor_np.shape)
  #     path1, path2 = torch.tensor(anchor_np), torch.tensor(positive_np)
  #     print(path1.shape)
  #     vector1 = classifier(path1)
  #     vector2 = classifier(path2)
  #     euclid_dis = LA.norm(vector1 - vector2)
  #     result['positive'].append(euclid_dis)
      
  #     path1, path2 = torch.tensor(anchor_np), torch.tensor(negative_np)
  #     vector1 = classifier(path1)
  #     vector2 = classifier(path2)
  #     euclid_dis = LA.norm(vector1 - vector2)
  #     result['negative'].append(euclid_dis)
  #     # break
  #   res_df = pd.DataFrame.from_dict(result)
  #   db_path = 'datacsv'
  #   res_df.to_csv(os.path.join(db_path, 'distance.csv'),index=False)
  #   print(res_df.describe())
  #   break
  cloth_model = TripletCombineInference(model_path=model_path, classifier_path=classifier_path)
  image1 = "last_val_data/anchor/62.jpg"
  image2 = "last_val_data/negative/62 (3).jpg"
  print(cloth_model.compare(image1, image2))
    