# import pandas as pd
# import os
# from sklearn.metrics import accuracy_score, classification_report
# from tqdm import tqdm
# from triplet_inference import TripletCombineInference

# csv_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv'

# df = pd.read_csv(os.path.join(csv_path, 'pair_triplet.csv'))
# im1_list = df.im1.to_list()
# im2_list = df.im2.to_list()
# label_list = df.label.to_list()

# model_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/'
# classifier_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/triplet_model/'
# file = open(os.path.join(csv_path, 'euclid.txt'), 'a')
# max_acc = 0
# chosen = 0

# for euclid in tqdm(range(30,50)):
#     threshold = euclid / 10
#     actuals, preds = [], []
#     cloth_model = TripletCombineInference(model_path=model_path, classifier_path=classifier_path)
#     for idx in tqdm(range(len(im1_list))):
#         path1, path2 = im1_list[idx], im2_list[idx]
#         actuals.append(label_list[idx])
        
#         distance, check = cloth_model.compare(path1, path2)
        
#         if check:
#             preds.append(1)
#         else:
#             preds.append(0)
        
#     acc = accuracy_score(actuals, preds)
#     print(classification_report(actuals, preds))
#     file.write(f"{threshold}: {acc}\n")
#     if acc > max_acc:
#         max_acc = acc
#         chosen = threshold
    
#     # if count > 10: break

# file.close()   
# print(chosen, max_acc)
    
from glob import glob
from itertools import combinations
from inference import Inference
from triplet_inference import TripletCombineInference
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import random
import pandas as pd

# dict_key = {}
# dict_csv = {
#     'im1': [],
#     'im2': [],
#     'label': []
# }
# for i in glob('/AIHCM/ComputerVision/hungtd/fashion-dataset/test_final/*.jpg'):
#     key = i.split('/')[-1].split('(')[0]
#     if key not in dict_key:
#         dict_key[key] = []
#     dict_key[key].append(i)
    
# for k in dict_key:
#     x = random.sample(list(combinations(dict_key[k], 2)), 10)
#     for _ in x:
#         dict_csv['im1'].append(_[0])
#         dict_csv['im2'].append(_[1])
#         dict_csv['label'].append(1)
#     count = 0
#     while True:
#         key_list = [_ for _ in dict_key.keys() if _ != k]
        
#         k_dif = random.choice(key_list)
#         dict_csv['im1'].append(random.choice(dict_key[k]))
#         dict_csv['im2'].append(random.choice(dict_key[k_dif]))
#         dict_csv['label'].append(0)
#         count += 1
#         if count >= 10:
#             break
#     # break

# df = pd.DataFrame.from_dict(dict_csv)
# # print(df)
# df.to_csv('/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv/test_final.csv')


model_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/crop_pytorch_model/'
classifier_path = '/AIHCM/ComputerVision/hungtd/fashion-dataset/triplet_model/'
df = pd.read_csv('/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv/test_final.csv')

im1_list = df.im1.to_list()
im2_list = df.im2.to_list()
label_list = df.label.to_list()



for testcase in ['not_triplet', 'triplet']:
    print(f'--------------{testcase}---------------')
    if testcase == 'triplet':
        cloth_model = TripletCombineInference(model_path=model_path, classifier_path=classifier_path)
    else:
        cloth_model = Inference(model_path=model_path)
    actuals, preds = [], []   
    for i in tqdm(range(len(label_list))):
        im1, im2 = im1_list[i], im2_list[i]
        actuals.append(label_list[i])
        check = cloth_model.compare(im1, im2)
        if check:
            preds.append(1)
        else:
            preds.append(0)
        # break
    print(classification_report(actuals,preds,digits=4))