# Cloth-Similarity-Model
Deep Learning model for feature extractions from cloth images and its application for:

* Similarity prediction

* Video clustering

## Table of contents
* [Dataset](#dataset)
* [Proposed architecture](#proposed-architecture)
* [Setup](#setup)
* [Similarity prediction](#similarity-prediction)
* [Video clustering](#video-clustering)


## Dataset: 

Dataset, which includes raw images and pretrained model, used in this project is stored in this google drive link: https://drive.google.com/file/d/10e2jCu5CS7VrgNPQBzsN9U0erd4Zwr5q/view?usp=sharing

* crop_images: dataset for training backbone model

* datacsv: dataset storing as csv

* last_val_data: dataset for triplet validation

* crop_pytorch_model: pretrained backbone model

* models: pretrained yolo crop model

* triplet_model: pretrained classifier model training by triplet_loss

## Proposed architecture

* Version 1: I call it "backbone model" because it will be re-used as the backbone for model version 2.0. This is the model architecture:

<p align="center">
  <img align="center" src="git_img/Cloth Model.png" />
</p>
<p align="center">
  <b>Figure 1:</b> Proposed Cloth-Similarity-Model v1.0
</p>

In detail, let's look closer to folder ./code:

* To generate dataset in csv format, run the following command: 
```bash
  python gen_data_csv.py
```

* To train model v1.0, run the following command: 
```bash
  python train.py
```

File train.py use 2 classes defined in `DataModule.py` to init DataLoader and class TorchVisionClassifierTrainer in `TorchVisionClassifierTrainer.py` to define model v1.0's architecture and model v1.0's trainer.

* To inference or test by categorical output with model v1.0, run the following command: 
```bash
  python inference.py
```

or

```bash
  python test.py
```

* To train model v2.0, run the following command: 
```bash
  python triplet_train.py
```

File `triplet_train.py` use 2 classes defined in `TripletDataModule.py` to init DataLoader and class TorchVisionClassifierTrainer in `TripletTrainer.py` to define model v2.0's architecture and model v2.0's trainer. In which, file `TripletTrainer.py` use class CombineModel in `CombineModel.py` to combine model v1.0's architecture with 3 

* To inference or test by categorical output with model v1.0, run the following command: 
```bash
  python inference.py
```

or

```bash
  python test.py
```


## Similarity prediction

## Video clustering

This task is per