# DongKyu Kim, Sungwon Kim
# CGML Midterm
# Professor Curro
#
# We are implementing:
# You Only Look Once: Unified, Real-Time Object Detection
# https://arxiv.org/pdf/1506.02640.pdf
#
# Specifically we are implementing the funtionality of the model such that
# we would reproduce something equivalent to Figure 6.
#
# The goal of the project is to build a model that predicts an object's label
# and the object's bounding box in an image based on the YOLO model, and
# Pascal VOC objetc detection dataset, which is further discussed in
# data_load.py

# YOLO v2 for reference
# https://arxiv.org/pdf/1612.08242.pdf

# Library and Function Imports
import tensorflow as tf 
import keras
from keras.models import Model
from data_load import load_data

# Data Load
ann_dir = '/home/dongkyu/VOC2012/Annotations/'
img_dir = '/home/dongkyu/VOC2012/JPEGImages/'
lab_dir = '/home/dongkyu/VOC2012/Labels/'
labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

image_h, image_w = 448, 448
out_h, out_w = 13,13
box = 5

train, val, test = load_data(ann_dir,img_dir,lab_dir,labels)

