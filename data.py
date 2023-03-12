import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import random
import os
from PIL import Image
from os import listdir
import math

def colorspace(image, Ycbcr=False, RGB=True, HSV=False):
    if Ycbcr:
        return cv2.cvtColor(image, cv2.cv.CV_BGR2YCrCb)
    elif RGB:
        return image
    elif HSV:
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    else:
        print("Have to choose one color space")
        return None

def normalizer(image):
    return cv2.normalize(np.array(image), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

def datapreprocess(image):
    image = colorspace(image)   ## convert color spaces
    image = normalizer(image)   ## normalize the image
    return image

class facedata():
    def __init__(self, truePath, falsePath, random=True):
        self.truePath = truePath
        self.falsePath = falsePath
        self.data = []
        self.random = random
        self.imgsize = 600
        self.datasize = 0

    def load(self):
        print("------------------------------------------------------")
        print("Loading data ...")
        for filename in os.listdir(self.truePath):
            image = Image.open(self.truePath+filename)
            image = datapreprocess(image)
            self.data.append({'img':image, 'label':0})
            self.datasize += 1

        for filename in os.listdir(self.falsePath):
            image = Image.open(self.falsePath+filename)
            image = datapreprocess(image)
            self.data.append({'img':image, 'label':1})
            self.datasize += 1
        print("Total size of the dataset:", len(self.data))
        

    def setup(self, portion):  
        ## portion is the percentage of the testing and training set
        if self.datasize == 0:
            print("No dataset loaded")
            return 
        self.trainsize = round(self.datasize * (1-portion))
        self.testsize = round(self.datasize * portion)
        print("Training datasize:", self.trainsize, "  Testing datasize:", self.testsize)
        print("------------------------------------------------------")
        picked = []
        while len(picked) <= self.testsize:
            randnum = random.randrange(0, self.datasize) 
            if randnum not in picked:
                picked.append(randnum)
                ## append the picked data into training data and testing data
        
            




def loaddata(truePath, falsePath):
    random_picker = False ## randomly pick pictures for testing
    dataholder = facedata(truePath, falsePath, random_picker)
    dataholder.load()  ## load true and false lable path data 

    random_seed = 0.2  ## determine the portion of the size fo testing dataset
    dataholder.setup(random_seed)  ## set up the training and testing set

