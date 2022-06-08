import time
import os
import torch
import json
import numpy as np
from PIL import Image
import cv2 
from torchvision import transforms

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from DataModule import DataModule


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

datamodule = DataModule(data_path='/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv', img_size = 224)
datamodule.setup()

# model_path = "/AIHCM/ComputerVision/hungtd/fashion-dataset/pytorch_model/"
model_path = "/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path + "best_model.pth", map_location=device)
config = json.load(open(model_path + "config.json", "r"))
model.eval()

output_article, output_color = {}, {}
def get_features(name, layer_name):
    def hook(model, input, output):
        if layer_name=='article':
            output_article[name] = output.detach()
        elif layer_name=='color':
            output_color[name] = output.detach()
    return hook
model.classifier_articleType[3].register_forward_hook(get_features(3, layer_name='article'))
model.classifier_baseColour[3].register_forward_hook(get_features(3, layer_name='color'))



# Test
train, test = datamodule.train_ds, datamodule.val_ds
id2articleType, id2baseColour = datamodule.id2articleType, datamodule.id2baseColour
articleType2id, baseColour2id = datamodule.articleType2id, datamodule.baseColour2id

start = time.time()
# --------------------------------- Get layer_name -----------------------------
def extract_features(img, img_size=224):
    '''
    Input: img can be a string or PIL image object
    Output: Tensors
    '''
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
        print(img.size)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(np.uint8(img)).convert('RGB')
    else:
        img = img
         
    val_trans = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    img = val_trans(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_article, pred_color = model(img)

        labels_article = list(config["articleType2id"].keys())
        labels_color = list(config["baseColour2id"].keys())
        softmax = torch.nn.Softmax(dim=0)
        
        probabilities_article = torch.mul(softmax(pred_article[0]), 100).tolist()
        probabilities_color = torch.mul(softmax(pred_color[0]), 100).tolist()
        
        prob_dict_article = dict(zip(labels_article, probabilities_article))
        prob_dict_color = dict(zip(labels_color, probabilities_color))
        
        max_article = sorted(prob_dict_article.values())[-1]
        max_color = sorted(prob_dict_color.values())[-1]
        
        max_key_article = list(prob_dict_article.keys())[list(prob_dict_article.values()).index(max_article)]
        max_key_color = list(prob_dict_color.keys())[list(prob_dict_color.values()).index(max_color)]
        print(max_key_article, max_key_color)
        return output_article[3].detach().cpu().numpy(), output_color[3].detach().cpu().numpy()
        # return torch.add(output_article[3], output_color[3]*4)

PATH1 = '/AIHCM/ComputerVision/hungtd/fashion-dataset/image_2022_04_21T04_41_03_951Z.png'
PATH2 = '/AIHCM/ComputerVision/hungtd/fashion-dataset/b.png'

# vector1, vector1_1 = extract_features(PATH1)
# vector2, vector2_1 = extract_features(PATH2)
# print((cosine_similarity(vector1, vector2) + cosine_similarity(vector1_1, vector2_1))/2)

# --------------------------------- Get layer_name -----------------------------
    
y_true_article = []
y_pred_article = []
y_true_color = []
y_pred_color = []

for idx in tqdm(range(len(test))):
    img = test[idx][0]
    i = config["id2articleType"][str(torch.IntTensor.item(test[idx][1][0]))]
    j = config["id2baseColour"][str(torch.IntTensor.item(test[idx][1][1]))]
    y_true_article.append(i)
    y_true_color.append(j)

    img = img.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_article, pred_color = model(img)

        labels_article = list(config["articleType2id"].keys())
        labels_color = list(config["baseColour2id"].keys())
        softmax = torch.nn.Softmax(dim=0)
        
        probabilities_article = torch.mul(softmax(pred_article[0]), 100).tolist()
        probabilities_color = torch.mul(softmax(pred_color[0]), 100).tolist()
        
        prob_dict_article = dict(zip(labels_article, probabilities_article))
        prob_dict_color = dict(zip(labels_color, probabilities_color))
        
        max_article = sorted(prob_dict_article.values())[-1]
        max_color = sorted(prob_dict_color.values())[-1]
        
        max_key_article = list(prob_dict_article.keys())[list(prob_dict_article.values()).index(max_article)]
        max_key_color = list(prob_dict_color.keys())[list(prob_dict_color.values()).index(max_color)]
        
        y_pred_article.append(max_key_article)
        y_pred_color.append(max_key_color)
        
print(classification_report(y_true_article, y_pred_article, digits=4))
print(classification_report(y_true_color, y_pred_color, digits=4))


print(f'Prediction time: {time.time() - start}')