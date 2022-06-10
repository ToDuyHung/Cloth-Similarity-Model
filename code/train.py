import os
from DataModule import DataModule
from TorchVisionClassifierTrainer import TorchVisionClassifierTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

datamodule = DataModule(data_path='datacsv', img_size = 224)
datamodule.setup()

# Test
train, test = datamodule.train_ds, datamodule.val_ds
id2articleType, id2baseColour = datamodule.id2articleType, datamodule.id2baseColour
articleType2id, baseColour2id = datamodule.articleType2id, datamodule.baseColour2id
print(len(test))
# print(test[0])
print(id2articleType)
print(id2baseColour)


trainer = TorchVisionClassifierTrainer(
	output_dir   = "crop_pytorch_model/",
	model_name   = 'resnet50',
	train      	 = train,
	test      	 = test,
	batch_size   = 32,
	max_epochs   = 500,
	id2articleType 	 = id2articleType,
	articleType2id 	 = articleType2id,
    id2baseColour 	 = id2baseColour,
	baseColour2id 	 = baseColour2id,
	lr=1e-3,
	use_soft_attention=False
)