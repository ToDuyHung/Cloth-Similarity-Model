import os
import json
import time
import glob

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import faiss   
from sklearn import preprocessing

from model_utils import crop_batch


class Inference:

  def __init__(self, model_path: str, threshold = 0.85, device=None):
    """
    🤗 Constructor for the image classifier trainer of TorchVision
    """
    self.img_size = 224
    self.resize_shape = (240,320)
    self.threshold = threshold
    self.db_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset'
    self.use_gpu = False
    self.img_transform = self.transform(self.img_size)
    if device == None:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
      self.device = device
    self.model_path = model_path if model_path.endswith("/") else model_path + "/"
    self.model = torch.load(self.model_path + "best_model.pth", map_location=self.device)
    self.model.eval()
    self.config = json.load(open(self.model_path + "config.json", "r"))
    self.output_article = {}
    self.output_color = {}
    self.model.classifier_articleType[3].register_forward_hook(self.get_features(3, layer_name='article'))
    self.model.classifier_baseColour[3].register_forward_hook(self.get_features(3, layer_name='color'))

  @staticmethod
  def transform(img_size):
    return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
  def get_features(self, name, layer_name):
      def hook(model, input, output):
          if layer_name=='article':
              self.output_article[name] = output.detach()
          elif layer_name=='color':
              self.output_color[name] = output.detach()
      return hook
  def extract_image(self, img, save_preview=False):
    """
    🤔 Predict from one image at the numpy array format or string format
    """
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(np.uint8(img)).convert('RGB')
    else:
        img = img
    img = img.resize(self.resize_shape)
    # img.save('a.jpg')
    img = self.img_transform(img)
    if save_preview:
      pil_img = transforms.ToPILImage()(img)
      pil_img.save("preview.jpg")
    
    img = torch.unsqueeze(img, 0).to(self.device)

    with torch.no_grad():
        self.model(img)
        return (self.output_article[3].detach().cpu().numpy() + self.output_color[3].detach().cpu().numpy())/2
  def compare(self, img1, img2):
    '''
    Return True if img1 and img2 are similar, otherwise False
    '''
    vector1 = self.extract_image(img1)
    vector2 = self.extract_image(img2)
    cos_sim = cosine_similarity(vector1, vector2)
    # print(f'Cosine similarity: {cos_sim}')
    if cos_sim > self.threshold:
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
  inference = Inference(model_path="crop_pytorch_model/")
  # inference.gen_db(dataset_path="Hume_data_full")
  path1 = 'test_imgs/1.png'
  path2 = 'test_imgs/2.png'
  img_dict = {0: cv2.resize(cv2.imread(path1), (240,320)) ,\
              1: cv2.resize(cv2.imread(path2), (240,320))}
  result_dict = crop_batch(img_dict)
  cv2.imwrite('test_imgs/test1.jpg', result_dict[0][1])
  cv2.imwrite('test_imgs/test2.jpg', result_dict[1][1])
  img1, img2 = result_dict[0][1], result_dict[1][1]
  # list_len, list_id, index = inference.load_db()
  # print(inference.identification(img_file= 'Hume_data_full/Đầm body lưới - D004 - Đen - S,M,L/6a38781c89a777f92eb6.jpg', list_len=list_len, list_id=list_id, index=index))
  print(inference.compare(img1, img2))
  