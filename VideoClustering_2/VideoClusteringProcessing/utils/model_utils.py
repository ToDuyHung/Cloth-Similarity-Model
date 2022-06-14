import logging
from typing import List, Dict
import os
from objects.singleton import Singleton
import json
import time
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import faiss   
from sklearn import preprocessing
import cv2

###################### Yolo model ###############################
from PIL import Image
from pathlib import Path
import torch
import sys

from config.config import Config

cwd = os.getcwd()
sys.path.append(cwd + '/model')

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, check_suffix, non_max_suppression, scale_coords, set_logging
from yolov5.utils.torch_utils import select_device

ROOT = "yolov5"

class CropModel:
    
    def __init__(self):
        self.config = Config()
        self.model_yolo_crop, self.stride, self.half, self.device = self.load_yolo_model(weights=self.config.crop_model_path, device='cpu' if not self.config.use_gpu else 'cuda')

    @torch.no_grad()
    def load_yolo_model(self, weights=ROOT + '/yolov5s.pt',
                        source=ROOT + '/data/images',  # file/dir/URL/glob, 0 for webcam
                        imgsz=640,  # inference size (pixels)
                        conf_thres=0.25,  # confidence threshold
                        iou_thres=0.45,  # NMS IOU threshold
                        max_det=1000,  # maximum detections per image
                        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                        view_img=False,  # show results
                        save_txt=False,  # save results to *.txt
                        save_conf=False,  # save confidences in --save-txt labels
                        save_crop=False,  # save cropped prediction boxes
                        nosave=False,  # do not save images/videos
                        classes=0,  # filter by class: --class 0, or --class 0 2 3
                        agnostic_nms=False,  # class-agnostic NMS
                        augment=False,  # augmented inference
                        visualize=False,  # visualize features
                        update=False,  # update all models
                        project=ROOT + '/runs/detect',  # save results to project/name
                        name='exp',  # save results to project/name
                        exist_ok=False,  # existing project/name ok, do not increment
                        line_thickness=3,  # bounding box thickness (pixels)
                        hide_labels=False,  # hide labels
                        hide_conf=False,  # hide confidences
                        half=False,  # use FP16 half-precision inference
                        dnn=False,  # use OpenCV DNN for ONNX inference
                        ):
        set_logging()
        device = select_device(device)
        half &= device.type != 'cpu'  
        w = str(weights[0] if isinstance(weights, list) else weights)
        classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
        check_suffix(w, suffixes)
        pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)
        stride, names = 64, [f'class{i}' for i in range(1000)] 
        if pt:
            model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
            stride = int(model.stride.max())
            names = model.module.names if hasattr(model, 'module') else model.names  
            if half:
                model.half()  
        for name in dir():
            if name not in ["model", "stride", "half", "device"]:
                del name
        return model, stride, half, device

    def choose_best_crop(self, X1, Y1, X2, Y2, confidences, crop_method=1):
        if len(confidences) == 0:
            return None, None, None, None, None
        list_index_filter_confidence = [i for i in range(len(confidences)) if confidences[i] >= self.config.crop_conf]
        if len(list_index_filter_confidence) == 0:
            list_area = [abs(X1[i] - X2[i]) * abs(Y1[i] - Y2[i]) for i in range(len(confidences))]
            max_area_index = list_area.index(max(list_area))
            return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[
                max_area_index]
        elif len(list_index_filter_confidence) == 1 and crop_method == 1:
            index = list_index_filter_confidence[0]
            return X1[index], Y1[index], X2[index], Y2[index], confidences[index]
        elif (len(list_index_filter_confidence) > 1) and crop_method == 1:
            list_area = [abs(X1[index] - X2[index]) * abs(Y1[index] - Y2[index]) for index in list_index_filter_confidence]
            max_area_index = list_index_filter_confidence[list_area.index(max(list_area))]
            return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[
                max_area_index]
        elif len(list_index_filter_confidence) >= 1 and crop_method == 2:
            index = confidences.index(max(confidences))
            return X1[index], Y1[index], X2[index], Y2[index], confidences[index]

    def batch_bbox(self, batch_img: List[np.array], conf_thres=0.2, img_size=416):
        pt, max_det, iou_thres = True, 1000, 0.45
        img_size = check_img_size(imgsz=416, s=self.stride)  
        images = []
        for img in batch_img:
            img = LoadImages(img, img_size=img_size, stride=self.stride, auto=pt)
            images.append(img)
        im0s_shape, img_shape = None, None
        for key, i in enumerate(images):
            for _, img, im0s, _ in i:
                im0s_shape = im0s.shape
                images.pop(key)
                images.insert(key, img)
        images = np.array(images)
        img_shape = images.shape
        images = torch.from_numpy(images).to(self.device)
        images = images.half() if self.half else images.float()  
        images /= 255.0
        preds = self.model_yolo_crop(images, augment=False, visualize=False)[0]
        preds = non_max_suppression(preds, conf_thres, iou_thres, classes=0, max_det=max_det)
        list_detections = {}
        for i, det in enumerate(preds): 
            list_detections[i] = []
            if len(det):
                det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0s_shape).round()
                for *xyxy, conf, _ in reversed(det):
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    list_detections[i].append({'bounding_box': [x1, y1, x2, y2], 'confidence': conf})
        for name in dir():
            if name != "list_detections":
                del name
        return list_detections

    def crop_batch(self, dict_img: Dict, alpha=0., crop_method=1.):
        batch_img = list(dict_img.values()) 
        list_detections = self.batch_bbox(batch_img, conf_thres=0.2, img_size=416)
        check_clothes = []
        for img_index, pred in list_detections.items():
            img = batch_img[img_index]
            h, w, c = img.shape
            X1, X2, Y1, Y2 = [], [], [], []
            confidences = []
            if len(pred) != 0:
                for detection in pred:
                    x1, y1, x2, y2 = detection["bounding_box"]
                    confidence = detection["confidence"]
                    X1.append(x1)
                    X2.append(x2)
                    Y1.append(y1)
                    Y2.append(y2)
                    confidences.append(confidence)
            else:
                check_clothes.append(False)
                continue
            X1, Y1, X2, Y2, confidence = self.choose_best_crop(X1, Y1, X2, Y2, confidences, crop_method)
            if X2:
                top_left_x, top_left_y, bot_right_x, bot_right_y = X1, Y1, X2, Y2
                top_left_x = int(max(top_left_x - alpha * (bot_right_x - top_left_x), 0))
                bot_right_x = int(min(bot_right_x + alpha * (bot_right_x - top_left_x), w))
                top_left_y = int(max(top_left_y - alpha * 2 * (bot_right_y - top_left_y), 0))
                bot_right_y = int(min(bot_right_y + alpha * 3 * (bot_right_y - top_left_y), h))
                img = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
            else:
                print("There is no bounding box...")
            batch_img[img_index] = img
            check_clothes.append(True)
        for index, value in enumerate(list(dict_img.keys())):
            dict_img[value] = [check_clothes[index], batch_img[index]]
        logging.info("CHECK CLOTH " + str(check_clothes))
        for name in dir():
            if name != "dict_img":
                del name
        return dict_img


class ClothModel:
    
    def __init__(self, device=None):
        """
            Constructor for the image classifier trainer of TorchVision
        """
        self.config = Config()
        self.crop_model = CropModel()
        self.img_size = 224
        self.resize_shape = (240,320)
        self.threshold = 0.9
        self.img_transform = self.transform(self.img_size)
        if device == None:
            self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.config.use_gpu) else "cpu")
        else:
            self.device = device
        sys.path.append('utils')
        self.model = torch.load(self.config.cloth_model_path + "/best_model.pth", map_location=self.device)
        self.model.eval()
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
            Predict from one image at the numpy array format or string format
        """
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(np.uint8(img)).convert('RGB')
        else:
            img = img
        img = img.resize(self.resize_shape)
        img = self.img_transform(img)
        if save_preview:
            pil_img = transforms.ToPILImage()(img)
            pil_img.save("preview.jpg")
        img = torch.unsqueeze(img, 0).to(self.device)
        with torch.no_grad():
            self.model(img)
            return (self.output_article[3].detach().cpu().numpy() + self.output_color[3].detach().cpu().numpy())/2

    def extract_image_batch(self, dict_img, save_preview=False):
        """
            Predict from one image at the numpy array format or string format
        """
        dict_img_2 = self.crop_model.crop_batch(dict_img, alpha = -0.05)   
        batch_img = []
        for i in dict_img_2.values():
            # a = np.random.randint(1000)
            # cv2.imwrite(f'output_video/{a}.jpg', i[1])
            batch_img.append(i[1])
        for index, img in enumerate(batch_img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize(self.resize_shape)
            img = self.img_transform(img)
            batch_img[index] = img
            if save_preview:
                pil_img = transforms.ToPILImage()(img)
                pil_img.save(f"output_video/preview_{index}.jpg")
        batch_img = torch.stack(batch_img, dim=0).to(self.device)
        with torch.no_grad():
            self.model(batch_img)
            features = (self.output_article[3].detach().cpu().numpy() + self.output_color[3].detach().cpu().numpy())/2
            for index, value in enumerate(list(dict_img.keys())):
                dict_img[value] = features[index]
            for name in dir():
                if name != "dict_img":
                    del name
            return dict_img
        
    def compare(self, img1, img2):
        '''
            Return True if img1 and img2 are similar, otherwise False
        '''
        vector1 = self.extract_image(img1)
        vector2 = self.extract_image(img2)
        cos_sim = cosine_similarity(vector1, vector2)
        if cos_sim > self.threshold:
            return cos_sim, True
        else:
            return cos_sim, False

    def gen_db(self, dataset_path:str):
        embedding = {}
        for dir in tqdm(os.listdir(dataset_path)):
            embedding[dir] = []
            for img_file in tqdm(glob.glob(os.path.join(dataset_path, dir, '*.jpg'))):
                vector = self.extract_image(img_file)
                vector = [list(vector[0].astype(dtype=np.float64))]
                embedding[dir].append(vector)
        with open(os.path.join(self.config.db_path, 'db.json'), 'w', encoding='utf-8') as f:
            json.dump(embedding, f)

    def load_db(self):
        with open(os.path.join(self.config.db_path, 'db.json'), 'r') as f:
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
                index = faiss.IndexFlatIP(d)
                if self.config.use_gpu:
                    device =  faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(device, 0, index)
                first_time = False
            for feature in v:
                list_feature.append(np.array(feature).astype('float32').reshape(1, -1))
        list_feature = np.concatenate(list_feature , axis=0)
        list_feature_new = preprocessing.normalize(list_feature, norm='l2')
        index.add(list_feature_new)
        return list_len, list_id, index

    def identification(self, img_file, list_len, list_id, index):
        res = None
        vectors = self.extract_image(img_file)
        max = 0
        for vector in vectors:
            xq = np.array(vector).astype('float32').reshape(1, -1)
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
            if distances[0][0] < self.threshold or distances[0][0] < max:
                continue
            max = distances[0][0]
            res = PID
        return res

