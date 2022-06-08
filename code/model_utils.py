from typing import List, Dict
from array import array
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import cv2
from tqdm import tqdm
import glob

###################### Yolo model ###############################
from PIL import Image
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import pandas as pd
import sys
cwd = os.getcwd()
sys.path.append('/AIHCM/ComputerVision/hungtd/fashion-dataset/models')
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import check_img_size, check_suffix, non_max_suppression, scale_coords, set_logging
from yolov5.utils.torch_utils import select_device
ROOT = "yolov5"

@torch.no_grad()
def load_yolo_model(weights=ROOT + '/yolov5s.pt',  # model.pt path(s)
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
    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    w = str(weights[0] if isinstance(weights, list) else weights)
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = torch.jit.load(w) if 'torchscript' in w else attempt_load(weights, map_location=device)
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
    return model, stride, half, device
model_yolo_crop, stride, half, device = load_yolo_model(weights=r"/AIHCM/ComputerVision/hungtd/fashion-dataset/models/yolov5/yolov5s.pt", device='cpu')

def bbox(source='test2', conf_thres=0.2, imgsz=416):
    pt, max_det, iou_thres = True, 1000, 0.45
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    bs = 1  # batch_size
    # Run inference
    if pt and device.type != 'cpu':
        model_yolo_crop(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model_yolo_crop.parameters())))  
    for _, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0 
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        # Inference
        if pt:
            pred = model_yolo_crop(img, augment=False, visualize=False)[0]
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0, max_det=max_det)
        # Process predictions
        list_detections = []
        for i, det in enumerate(pred):  # per image
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, _ in reversed(det):
                    x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    list_detections.append({'bounding_box': [x1,y1,x2,y2], 'confidence': conf})
                    # img = cv2.rectangle(im0s,(x1,y1),(x2,y2),color = [255,0,0], thickness=3)
                    # img = cv2.putText(img,str(conf),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                    # cv2.imshow('fds',img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
    return list_detections
##################################################################

def choose_best_crop(X1, Y1, X2, Y2, confidences, confidence_threshold = 0.92, crop_method=1):
    if (len(confidences) == 0):
        return None, None, None, None, None
    list_index_filter_confidence = [i for i in range(len(confidences)) if confidences[i] >= confidence_threshold]
    # neu khong co bbox nao co score > confidence_threshold => chon bbox co area max
    if (len(list_index_filter_confidence) == 0):
        list_area = [abs(X1[i] - X2[i]) * abs(Y1[i] - Y2[i]) for i in range(len(confidences))]
        max_area_index = list_area.index(max(list_area))
        return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[max_area_index]
    # neu co 1 bbox => chon luon
    elif (len(list_index_filter_confidence) == 1) and crop_method==1:
        index = list_index_filter_confidence[0]
        return X1[index], Y1[index], X2[index], Y2[index], confidences[index]
    # neu co nhieu hon 1 bbox thoa man => chon bbox co area max
    elif (len(list_index_filter_confidence) > 1) and crop_method==1:
        list_area = [abs(X1[index] - X2[index]) * abs(Y1[index] - Y2[index]) for index in list_index_filter_confidence]
        max_area_index = list_index_filter_confidence[list_area.index(max(list_area))]
        return X1[max_area_index], Y1[max_area_index], X2[max_area_index], Y2[max_area_index], confidences[max_area_index]
    elif len(list_index_filter_confidence)>=1 and crop_method==2:
        index = confidences.index(max(confidences))
        return X1[index], Y1[index], X2[index], Y2[index], confidences[index]

def crop(path, alpha=0, crop_method=1):
    try:
        pil_image = Image.open(path).convert('RGB') 
        open_cv_image  = np.array(pil_image)
        img = open_cv_image[:, :, ::-1]
    except:
        img = path             # if path is image array
    h,w,c = img.shape
    X1, X2, Y1, Y2 = [],[],[],[]
    confidences = []
    list_detections = bbox(source = path, conf_thres=0.2, imgsz=416)
    if len(list_detections) != 0 :
        for detection in list_detections:
            x1, y1, x2, y2 = detection["bounding_box"]
            confidence = detection["confidence"]
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)
            confidences.append(confidence)
    else:
        # add_or_not = False
        # print("Nothing detected...")
        return False, img
    X1, Y1, X2, Y2, confidence = choose_best_crop(X1, Y1, X2, Y2, confidences, crop_method)
        
    if X2:       
        top_left_x, top_left_y, bot_right_x, bot_right_y = X1, Y1, X2, Y2
        top_left_x  = int(max(top_left_x-alpha*(bot_right_x-top_left_x), 0))
        bot_right_x = int(min(bot_right_x+alpha*(bot_right_x-top_left_x), w))
        top_left_y  = int(max(top_left_y-alpha*2*(bot_right_y-top_left_y), 0))
        bot_right_y = int(min(bot_right_y+alpha*3*(bot_right_y-top_left_y), h))
        # img_draw = img.copy()
        # cv2.rectangle(img_draw, (top_left_x, top_left_y), (bot_right_x, bot_right_y), (255, 0, 0), 2)
        # cv2.imshow('fds',img_draw)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # folder = os.path.basename(os.path.dirname(path))
        # file_name = os.path.basename(path)
        # if not os.path.isdir("image_crop_drawing_test2/" + folder):
        #     os.mkdir("image_crop_drawing_test2/" + folder)
        # cv2.imwrite(r'C:\Users\TMT\Desktop\32.jpg', img_draw)
        # cv2.imwrite("ds.jpg", img_draw)
        img = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
    else:
        # add_or_not = False
        # print(path)
        print("There is no bounding box...")
    return True, img

def batch_bbox(batch_img:List[np.array], conf_thres=0.2, imgsz=416):
    pt, max_det, iou_thres = True, 1000, 0.45
    imgsz = check_img_size(imgsz=416, s=stride)  # check image size
    images = []
    for img in batch_img:
        img = LoadImages(img, img_size=imgsz, stride=stride, auto=pt)
        images.append(img)
    images_copy = images.copy()
    images = []
    im0s_shape, img_shape = None, None
    for i in images_copy:    
        for _, img, im0s, _ in i:
            im0s_shape = im0s.shape
            images.append(img)
    images = np.array(images)
    img_shape = images.shape
    images = torch.from_numpy(images).to(device)
    images = images.half() if half else images.float()  # uint8 to fp16/32
    images /= 255.0 
    preds = model_yolo_crop(images, augment=False, visualize=False)[0]
    preds = non_max_suppression(preds, conf_thres, iou_thres, classes=0, max_det=max_det)
    list_detections = {}
    for i, det in enumerate(preds):  # per image
        list_detections[i] = []
        if len(det):
            det[:, :4] = scale_coords(img_shape[2:], det[:, :4], im0s_shape).round()
            for *xyxy, conf, _ in reversed(det):
                x1,y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                list_detections[i].append({'bounding_box': [x1,y1,x2,y2], 'confidence': conf})
    return list_detections

def crop_batch(dict_img:Dict, alpha=0, crop_method=1):
    batch_img = list(dict_img.values())    
    list_detections = batch_bbox(batch_img, conf_thres=0.2, imgsz=416)
    check_clothes = []
    for img_index, pred in list_detections.items():
        img = batch_img[img_index]
        h,w,c = img.shape
        X1, X2, Y1, Y2 = [],[],[],[]
        confidences = []
        if len(pred) != 0 :
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
        X1, Y1, X2, Y2, confidence = choose_best_crop(X1, Y1, X2, Y2, confidences, crop_method)
            
        if X2:       
            top_left_x, top_left_y, bot_right_x, bot_right_y = X1, Y1, X2, Y2
            top_left_x  = int(max(top_left_x-alpha*(bot_right_x-top_left_x), 0))
            bot_right_x = int(min(bot_right_x+alpha*(bot_right_x-top_left_x), w))
            top_left_y  = int(max(top_left_y-alpha*2*(bot_right_y-top_left_y), 0))
            bot_right_y = int(min(bot_right_y+alpha*3*(bot_right_y-top_left_y), h))
            img = img[top_left_y:bot_right_y, top_left_x:bot_right_x]
        else:
            print("There is no bounding box...")
        batch_img[img_index] = img
        check_clothes.append(True)
    for index, value in enumerate(list(dict_img.keys())):
        dict_img[value] = [check_clothes[index], batch_img[index]]
    # print(check_clothes)
    return dict_img