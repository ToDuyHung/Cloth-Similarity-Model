import os
import json
import math
import random
import argparse
from pathlib import Path
import multiprocessing as mp
from datetime import datetime
from collections import Counter

import torch
import torchmetrics
import torch.nn as nn
import numpy as np
from torchmetrics import Accuracy
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tqdm import tqdm
from tabulate import tabulate
from PIL import Image, ImageEnhance
from collections import Counter
from torch.utils.data.sampler import WeightedRandomSampler
from CombineModel import TripletCombineModel
# from SoftAttention import SoftAttention, AttentionModel

def NormalizeData(data):
  return (data - np.min(data)) / (np.max(data) - np.min(data))

class TripletTrainer:

  """
  🤗 Constructor for the image classifier trainer
  """
  def __init__(
    self,
    model_name        :str,
    train             :torch.utils.data.Dataset,
    test              :torch.utils.data.Dataset,
    output_dir    = "./models",
    max_epochs    = 1,
    workers       = mp.cpu_count(),
    batch_size    = 8,
    lr            = 2e-5,
    weight_decay  = 0.01,
    momentum      = 0.9,
    gamma         = 0.96,
    pretrained    = True,
    force_cpu     = False,
    requires_grad = False, # True: Only last layer of the classifier are updated
    load_best_model_at_end = True,
    classification_report_digits = 4,
    parallelized = False,
    lite = False,
    use_soft_attention = False,
  ):

    self.train             = train
    self.test              = test
    self.output_dir        = output_dir if output_dir.endswith("/") else output_dir + "/"
    self.lr                = lr
    self.weight_decay      = weight_decay
    self.momentum          = momentum
    self.gamma             = gamma
    self.batch_size        = batch_size
    self.max_epochs        = max_epochs
    self.workers           = workers
    self.model_name        = model_name
    self.pretrained        = pretrained
    self.best_acc          = 0
    self.best_loss         = 999999
    self.best_path         = ""
    self.logs_path         = self.output_dir + "logs/"
    self.config_path       = self.output_dir + "config.json"
    self.current_date      = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    self.config            = {}
    self.requires_grad     = requires_grad
    self.load_best_model_at_end = load_best_model_at_end
    self.classification_report_digits = classification_report_digits
    self.parallelized = parallelized
    self.lite = lite
    self.use_soft_attention = use_soft_attention
    self.triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)

    self.tensor_board = SummaryWriter()

    self.data_loader_train = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.workers)
    self.data_loader_test = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    self.image_example = next(iter(self.data_loader_train))[0]

    # Processing device (CPU / GPU)
    if force_cpu == True:
      self.device = "cpu"
    else:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the metric
    self.metric = torchmetrics.Accuracy()
    
    # Open the logs files
    self.__openLogs()
    self.__openLogsLossAcc()

    # Load the model from the TorchVision Models Zoo 
    if pretrained:
      print("Load the pre-trained model " + self.model_name)
      self.model = models.__dict__[self.model_name](pretrained=True)
    else:
      print("Load the model " + self.model_name)
      self.model = models.__dict__[self.model_name]()

    print(self.model)
    
    # """
    # 🏗️ Build the model
    # """
    if self.model_name.startswith('resnet'):
        # self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes) # uncomment for 1 output
        self.config["hidden_size"] = self.model.fc.in_features

    if self.parallelized == True:
        self.model = torch.nn.DataParallel(self.model)
    
    num_hidden = 1024
    classifier = nn.Sequential(
        nn.LazyLinear(num_hidden),
        # nn.BatchNorm1d(num_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden, 256),
        nn.Sigmoid()
    )
    
    self.model = TripletCombineModel(classifier, freeze_conv=False)
    self.model = self.model.to(self.device)

    if self.requires_grad:
      for param in self.model.parameters():
        param.requires_grad = False

    archi = ""
    archi += "="*50 + "\n"
    archi += "Model architecture:" + "\n"
    archi += "="*50 + "\n"
    archi += str(self.model) + "\n"
    archi += "="*50 + "\n"
    print(archi)
    
    # Write in logs
    self.__openLogs()
    self.logs_file.write(archi + "\n") 
    self.logs_file.close() 

    self.criterion = nn.TripletMarginLoss(margin=0.5, p=2).to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    # self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
    self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)

    self.config["architectures"] = [self.model_name]
    
    with open(self.config_path, 'w') as json_file:
      json.dump(self.config, json_file, indent=4)

    """
    ⚙️ Train the given model on the dataset
    """
    print("Start Training!")
    print('CUDA AVAILABLE: ', torch.cuda.is_available())
    print(self.device)
    self.training()

    if self.load_best_model_at_end:
      print("\033[95mBest model loaded!\033[0m")
      self.model = torch.load(self.output_dir + 'best_model.pth', map_location=self.device)

    # Close the logs file
    self.logs_file.close() 
    self.logs_loss_train.close()
    self.logs_acc_train.close()

    if self.lite == True:
      lite_path = output_dir + "model.ptl"
      print(lite_path)
      traced_script_module = torch.jit.trace(self.model.to("cpu"), self.image_example.to("cpu"))
      traced_script_module_optimized = optimize_for_mobile(traced_script_module)
      traced_script_module_optimized._save_for_lite_interpreter(lite_path)
      print("Lite exported!")

  """
  📜 Open the logs file
  """
  def __openLogs(self):

    # Check if the directory already exist
    os.makedirs(self.logs_path, exist_ok=True)

    # Open the logs file
    self.logs_file = open(self.logs_path + "logs_" + self.current_date + ".txt", "a")

  """
  📜 Open the logs file for loss and accuracy
  """
  def __openLogsLossAcc(self):

    if not self.logs_path.endswith("/"):
      self.logs_path += "/"

    # Check if the directory already exist
    os.makedirs(self.logs_path, exist_ok=True)

    # Open the logs file
    self.logs_loss_train = open(self.logs_path + "train_logs_loss_" + self.current_date + ".txt", "a")
    self.logs_acc_train = open(self.logs_path + "train_logs_acc_" + self.current_date + ".txt", "a")
    self.logs_acc_test = open(self.logs_path + "test_logs_acc_" + self.current_date + ".txt", "a")
  
  """
  👩‍🎓 Training phase
  """
  def training(self):

    for epoch in tqdm(range(self.max_epochs)):

      self.__openLogs()

      # Train the epoch
      batches_loss = self.compute_batches(epoch)

      # Evaluate on validation dataset
      avg_loss = self.evaluate()

      os.makedirs(self.output_dir, exist_ok=True)

      if avg_loss < self.best_loss:
        filename = self.output_dir + 'best_model.pth'
        self.best_path = filename
      else:
        filename = self.output_dir + 'last_model.pth'

      self.best_loss = min(avg_loss, self.best_loss)

      torch.save(self.model, filename)
      
      saved_at = "Model saved at: \033[93m" + filename + "\033[0m"
      print(saved_at)
      best_model_path = "\033[93m[" + self.model_name + "]\033[0m Best model saved at: \033[93m" + self.best_path + "\033[0m" + " - Loss " + "{:.4f}".format(self.best_loss)
      print(best_model_path)

      # self.logs_file.write(f1_score + "\n")
      self.logs_file.write(saved_at + "\n")
      self.logs_file.write(best_model_path + "\n")
      self.logs_file.close()

      self.tensor_board.add_scalar('Loss/train', batches_loss, epoch)
      self.logs_loss_train.write(str(epoch) + "," + str(batches_loss.item()) + "\n")

      self.tensor_board.add_scalar('Loss/test', avg_loss, epoch)
      self.logs_acc_test.write(str(epoch) + "," + str(avg_loss) + "\n")

  """
  🗃️ Compute epoch batches
  """
  def compute_batches(self, epoch):

    # Switch to train mode
    self.model.train()

    sum_loss = 0

    # For each batch
    for i, (input) in tqdm(enumerate(self.data_loader_train)):
      anchor, positive, negative = input
    
      anchor = anchor.to(self.device)
      positive = positive.to(self.device)
      negative = negative.to(self.device)
      
      anchor_var = torch.autograd.Variable(anchor)
      positive_var = torch.autograd.Variable(positive)
      negative_var = torch.autograd.Variable(negative)
      
      output_anchor = self.model(anchor_var)
      output_positive = self.model(positive_var)
      output_negative = self.model(negative_var)
      
      # output_anchor = (output_anchor - output_anchor.min())/(output_anchor.max() - output_anchor.min())
      # output_positive = (output_positive - output_positive.min())/(output_positive.max() - output_positive.min())
      # output_negative = (output_negative - output_negative.min())/(output_negative.max() - output_negative.min())

      # compute output
      loss = self.criterion(output_anchor, output_positive, output_negative)
      sum_loss += loss

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if i % (len(self.data_loader_train) / 10) == 0:
        log_line = "[Epoch " + str(epoch) + "], [Batch " + str(i) + " / " + str(self.max_epochs) + "], [Loss " + str(loss.item()) + "]"
        self.logs_file.write(log_line + "\n")

    self.scheduler.step()

    # Compute accuracy
    avg_loss = sum_loss / len(self.data_loader_train)
    
    print(f'Avg loss: {avg_loss}')

    return avg_loss

  """
  🧪 Evaluate the performances of the system of the test sub-dataset
  """
  def evaluate(self):

    with torch.no_grad():

      # Switch to evaluate mode
      self.model.eval()
      
      sum_loss = 0

      for i, (input) in enumerate(self.data_loader_test):
        anchor, positive, negative = input
        
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        
        # input_var = torch.autograd.Variable(input, volatile=True)
        output_anchor = self.model(anchor)
        output_positive = self.model(positive)
        output_negative = self.model(negative)
        
        # output_anchor = (output_anchor - output_anchor.min())/(output_anchor.max() - output_anchor.min())
        # output_positive = (output_positive - output_positive.min())/(output_positive.max() - output_positive.min())
        # output_negative = (output_negative - output_negative.min())/(output_negative.max() - output_negative.min())
        
        loss = self.criterion(output_anchor, output_positive, output_negative)
        sum_loss += loss
      avg_loss = sum_loss / len(self.data_loader_test)
      return avg_loss