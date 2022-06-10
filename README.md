# Cloth-Similarity-Model
Deep Learning model for feature extractions from cloth images and its application for similarity prediction

## Table of contents
* [Dataset](#dataset)
* [Proposed architecture](#proposed-architecture)
* [Setup](#setup)

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

