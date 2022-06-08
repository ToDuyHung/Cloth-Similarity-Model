import os
import json
import multiprocessing as mp
from datetime import datetime

import torch
import torchmetrics
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from tqdm import tqdm
from tabulate import tabulate
from PIL import Image, ImageEnhance
from torch.utils.data.sampler import WeightedRandomSampler
from CombineModel import CombineModel
# from SoftAttention import SoftAttention, AttentionModel

class TorchVisionClassifierTrainer:

  """
  🤗 Constructor for the image classifier trainer
  """
  def __init__(
    self,
    id2articleType,
	articleType2id,
    id2baseColour,
	baseColour2id,
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
    self.id2articleType    = id2articleType
    self.articleType2id    = articleType2id
    self.id2baseColour     = id2baseColour
    self.baseColour2id     = baseColour2id
    self.best_acc          = 0
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

    self.tensor_board = SummaryWriter()

    # ----- Add sampler weights-----

    
    sampler_weights = torch.zeros(len(self.train))
    class_sampler = self.train.baseColour_counts()
    for idx, label in enumerate(self.train.baseColour):
        sampler_weights[idx] = class_sampler[label]

    print(sampler_weights.shape)

    sampler_weights = 1000. / sampler_weights
    sampler = WeightedRandomSampler(sampler_weights.type('torch.DoubleTensor'), len(sampler_weights))
    self.data_loader_train = torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.workers, sampler=sampler)
    self.data_loader_test = torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

    self.image_example = next(iter(self.data_loader_train))[0]

    # Processing device (CPU / GPU)
    if force_cpu == True:
      self.device = "cpu"
    else:
      self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup the metric
    self.metric = torchmetrics.Accuracy()

    self.num_articleType = len(self.id2articleType.keys())
    self.num_baseColour = len(self.id2baseColour.keys())
    self.config["num_classes"] = self.num_articleType
    self.config["baseColour"] = self.num_baseColour
    
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
      
    # --------------------------- Add 2 outputs ----------------------------------
    img_block = torch.nn.Sequential(*(list(self.model.children())[:-1])) 
    num_hidden = 1024
    classifier_articleType = nn.Sequential(
        nn.LazyLinear(num_hidden),
        nn.BatchNorm1d(num_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden, self.num_articleType)
    )
    classifier_baseColour = nn.Sequential(
        nn.LazyLinear(num_hidden),
        nn.BatchNorm1d(num_hidden),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_hidden, self.num_baseColour)
    )
    self.model = CombineModel(img_block, classifier_articleType, classifier_baseColour, freeze_conv=False)
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

    # weight = torch.Tensor([3.0,3.0,1.0])
    self.criterion = nn.CrossEntropyLoss().to(self.device)
    self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    # self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
    self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma)

    self.config["id2articleType"] = self.id2articleType
    self.config["articleType2id"] = self.articleType2id
    self.config["id2baseColour"] = self.id2baseColour
    self.config["baseColour2id"] = self.baseColour2id
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

    # self.evaluate_f1_score()

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
  🧪 Evaluate the performances of the system of the test sub-dataset given a f1-score
  """
#   def evaluate_f1_score(self):

#     # Get the hypothesis and predictions
#     all_target, all_preds = self.evaluate()

#     table = metrics.classification_report(
#       all_target,
#       all_preds,
#       labels = [int(a) for a in list(self.ids2labels.keys())],
#       target_names = list(self.labels2ids.keys()),
#       zero_division = 0,
#       digits=self.classification_report_digits,
#     )
#     print(table)

#     # Write logs
#     self.__openLogs()
#     self.logs_file.write(table + "\n")
#     self.logs_file.close()

#     print("Logs saved at: \033[93m" + self.logs_path + "\033[0m")
#     print("\033[93m[" + self.model_name + "]\033[0m Best model saved at: \033[93m" + self.best_path + "\033[0m" + " - Accuracy " + "{:.2f}".format(self.best_acc*100))

#     return all_target, all_preds
  
  """
  👩‍🎓 Training phase
  """
  def training(self):

    for epoch in tqdm(range(self.max_epochs)):

      self.__openLogs()

      # Train the epoch
      batches_acc, batches_loss = self.compute_batches(epoch)

      # Evaluate on validation dataset
      all_targets_articleType, all_predictions_articleType, all_targets_baseColour, all_predictions_baseColour = self.evaluate()
      articleType_acc = accuracy_score(all_targets_articleType, all_predictions_articleType)
      baseColour_acc = accuracy_score(all_targets_baseColour, all_predictions_baseColour)
      total_acc = (articleType_acc + baseColour_acc) / 2

      # f1_score = classification_report(
      #   all_targets,
      #   all_predictions,
      #   target_names=list(self.ids2labels.values()),
      #   digits=self.classification_report_digits
      # )
      # print(f1_score)

      os.makedirs(self.output_dir, exist_ok=True)

      if total_acc > self.best_acc:
        filename = self.output_dir + 'best_model.pth'
        self.best_path = filename
      else:
        filename = self.output_dir + 'last_model.pth'

      self.best_acc = max(total_acc, self.best_acc)

      torch.save(self.model, filename)
      saved_at = "Model saved at: \033[93m" + filename + "\033[0m"
      print(saved_at)
      best_model_path = "\033[93m[" + self.model_name + "]\033[0m Best model saved at: \033[93m" + self.best_path + "\033[0m" + " - Accuracy " + "{:.2f}".format(self.best_acc*100) + "%"
      print(best_model_path)

      # self.logs_file.write(f1_score + "\n")
      self.logs_file.write(saved_at + "\n")
      self.logs_file.write(best_model_path + "\n")
      self.logs_file.close()

      self.tensor_board.add_scalar('Loss/train', batches_loss, epoch)
      self.logs_loss_train.write(str(epoch) + "," + str(batches_loss.item()) + "\n")

      self.tensor_board.add_scalar('Accuracy/train', batches_acc, epoch)
      self.logs_acc_train.write(str(epoch) + "," + str(batches_acc) + "\n")

      self.tensor_board.add_scalar('Accuracy/test', total_acc, epoch)
      self.logs_acc_test.write(str(epoch) + "," + str(total_acc) + "\n")

  """
  🗃️ Compute epoch batches
  """
  def compute_batches(self, epoch):

    # Switch to train mode
    self.model.train()

    sum_loss = 0

    all_preds_articleType = []
    all_targets_articleType = []
    all_preds_baseColour = []
    all_targets_baseColour = []

    # For each batch
    for i, (input, target) in tqdm(enumerate(self.data_loader_train)):
      output_articleType, output_baseColour = None, None
    
      input = input.to(self.device)
      # print(11111111111111111111111111111111111111111, input.shape)
      input_var = torch.autograd.Variable(input)
      # print(2222222222222222222222222222222222222222, input_var.shape)
      output_articleType, output_baseColour = self.model(input_var)
      target_articleType, target_baseColour = target
      target_articleType = target_articleType.to(self.device)
      target_baseColour = target_baseColour.to(self.device)
      target_articleType_var = torch.autograd.Variable(target_articleType)
      target_baseColour_var = torch.autograd.Variable(target_baseColour)

      # compute output
      loss_articleType = self.criterion(output_articleType, target_articleType_var)
      loss_baseColour = self.criterion(output_baseColour, target_baseColour_var)
      loss = (loss_articleType + loss_baseColour * 1.5) / 2

      sum_loss += loss
      all_preds_articleType.extend(torch.max(output_articleType, 1)[1].cpu().detach().numpy())
      all_targets_articleType.extend(target_articleType_var.cpu().detach().numpy())
      
      all_preds_baseColour.extend(torch.max(output_baseColour, 1)[1].cpu().detach().numpy())
      all_targets_baseColour.extend(target_baseColour_var.cpu().detach().numpy())

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      if i % (len(self.data_loader_train) / 10) == 0:
        log_line = "[Epoch " + str(epoch) + "], [Batch " + str(i) + " / " + str(self.max_epochs) + "], [Loss " + str(loss_articleType.item()) + "]"
        self.logs_file.write(log_line + "\n")
        log_line = "[Epoch " + str(epoch) + "], [Batch " + str(i) + " / " + str(self.max_epochs) + "], [Loss " + str(loss_baseColour.item()) + "]"
        self.logs_file.write(log_line + "\n")

    self.scheduler.step()

    # Compute accuracy
    articleType_acc = accuracy_score(all_targets_articleType, all_preds_articleType)
    baseColour_acc = accuracy_score(all_targets_baseColour, all_preds_baseColour)
    total_acc = (articleType_acc + baseColour_acc) / 2
    avg_loss = sum_loss / len(self.data_loader_train)
    
    print(f'articleType loss: {loss_articleType}, articleType accuracy: {articleType_acc}')
    print(f'baseColour loss: {loss_baseColour}, baseColour accuracy: {baseColour_acc}')
    print(f'Avg loss: {avg_loss}, Avg accuracy: {total_acc}')

    return total_acc, avg_loss

  """
  🧪 Evaluate the performances of the system of the test sub-dataset
  """
  def evaluate(self):

    with torch.no_grad():

      all_predictions_articleType = []
      all_targets_articleType = []
      all_predictions_baseColour = []
      all_targets_baseColour = []

      # Switch to evaluate mode
      self.model.eval()

      for i, (input, target) in enumerate(self.data_loader_test):
        output_articleType, output_baseColour = None, None
        target_articleType, target_baseColour = target
        
        input = input.to(self.device)
        # input_var = torch.autograd.Variable(input, volatile=True)
        output_articleType, output_baseColour = self.model(input)
        output_articleType = output_articleType.cpu().data.numpy()
        output_baseColour = output_baseColour.cpu().data.numpy()


        output_articleType = [o.argmax() for o in output_articleType]
        output_baseColour = [o.argmax() for o in output_baseColour]

        all_targets_articleType.extend(target_articleType.tolist())
        all_predictions_articleType.extend(output_articleType)
        all_targets_baseColour.extend(target_baseColour.tolist())
        all_predictions_baseColour.extend(output_baseColour)

      return all_targets_articleType, all_predictions_articleType, all_targets_baseColour, all_predictions_baseColour