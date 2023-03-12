import torch
from matplotlib import pyplot as plt
import os
import numpy as np
from data import loaddata

## set to run on GPU or CPU
print("Check if GPU is available:",torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
Load image dataset
Kaggle datset:
https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection/code?resource=download
Download the dataset and save it in the same directory as the train file(rename it to dataset)
'''
## Load dataset
real_datapath = "./dataset/real_and_fake_face/training_fake/"
fake_datapath = "./dataset/real_and_fake_face/training_real/"
loaddata(real_datapath, fake_datapath)

## train model

## save model

## test model


