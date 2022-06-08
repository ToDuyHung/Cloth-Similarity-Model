import os
from TripletDataModule import TripletDataModule
from TripletTrainer import TripletTrainer

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

datamodule = TripletDataModule(data_path='/AIHCM/ComputerVision/hungtd/fashion-dataset/datacsv', img_size = 224)
datamodule.setup()

train, test = datamodule.train_ds, datamodule.val_ds
print(len(train))
print(train[0][0].shape, train[0][1].shape)

trainer = TripletTrainer(
	output_dir   = "/AIHCM/ComputerVision/hungtd/fashion-dataset/triplet_gpu/",
	model_name   = 'resnet50',
	train      	 = train,
	test      	 = test,
	batch_size   = 32,
	max_epochs   = 500,
	lr=5e-4,
	use_soft_attention=False
)