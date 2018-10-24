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
#
# YOLO v2 for reference
# https://arxiv.org/pdf/1612.08242.pdf
#
# YOLO v3 for reference
# https://arxiv.org/pdf/1804.02767.pdf
#
# YOLOv3 uses 'Darknet-53' structure, which has 53 layers. We decided to start
# with YOLOv2's architecture which has 19 layers to make sure our thing works
# The original YOLO paper had a structure that has 26 total layers, and we 
# thought it will take approximately same time as YOLOv2's structure.


# Library and Function Imports
import tensorflow as tf 
import numpy as np
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten, Reshape
from keras.layers import BatchNormalization, LeakyReLU, Lambda, Input, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import LearningRateScheduler
from data_load import load_data, DataGenerator
#from data_infer import pretty_picture

# Data Load
ann_dir = '/home/dongkyu/VOC2012/Annotations/'
img_dir = '/home/dongkyu/VOC2012/JPEGImages/'
lab_dir = '/home/dongkyu/VOC2012/Labels/'
labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

train, val, test = load_data(ann_dir,img_dir,lab_dir,labels)

# Model Functions

def convconv(filters,size,strides,input):
	x = Conv2D(filters,size,strides = strides,padding='same')(input)
	x = BatchNormalization()(x)
	x = LeakyReLU(0.1)(x)
	return x	

def convblock1(filters,size,strides,input):
	x = convconv(filters,size,strides,input)
	x = MaxPooling2D(strides=2)(x)
	return x

def convblock2(filters,input):
	x = convconv(filters,1,1,input)
	x = convconv(filters*2,3,1,x)
	return x

def lr_adaptive(self,epoch):
    lr = 1e-3
    if epoch > 15:
    	lr = 1e-2
    if epoch > 75:
        lr = 1e-3
    if epoch > 105:
        lr = 1e-4
    return lr

def multipartloss(y,y_hat):
	coord = 5.
	noobj = 0.5

	l = coord*tf.reduce_sum(np.multiply(np.power((y_hat[:,:,:,0:4]-y[:,:,:,0:4]),2),y[:,:,:,5]))
	l += coord*tf.reduce_sum(np.multiply(np.power((y_hat[:,:,:,5:9]-y[:,:,:,0:4]),2),y[:,:,:,5]))
	l+=tf.reduce_sum(np.multiply(np.power(y_hat[:,:,:,4]-y[:,:,:,4],2),y[:,:,:,5]))
	l+=noobj*tf.reduce_sum(np.multiply(np.power(y_hat[:,:,:,9]-y[:,:,:,4],2),y[:,:,:,6]))
	l+=tf.reduce_sum(np.multiply(np.power((y_hat[:,:,:,10:30]-y[:,:,:,10:30]),2),y[:,:,:,5]))
	
	return l

# Model
#
# We obtained details about the yolov2 structure on
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
#

# Parameters
I_h = 448
S = 7
B = 2
C = len(labels)
out = B*5+C
learing_rate = 1e-3
epochs = 135
batch_size = 64

# YOLOv2 Model
input = Input(shape=(I_h,I_h,3))
x = convblock1(64,7,2,input)
x = convblock1(192,3,1,x)
x = convblock2(128,x)
x = convblock2(256,x)
x = MaxPooling2D(strides=2)(x)
for i in range(0,4):
	x = convblock2(256,x)
x = convblock2(512,x)
x = MaxPooling2D(strides=2)(x)
x = convblock2(512,x)
x = convblock2(512,x)
x = convconv(1024,3,1,x)
x = convconv(1024,3,2,x)
x = convconv(1024,3,1,x)
x = convconv(1024,3,1,x)
x = Flatten()(x)
x = Dense(256)(x)
x = Dense(4096)(x)
x = LeakyReLU(0.1)(x)
x = Dropout(0.5)(x)
x = Dense(S*S*out)(x)
y = Reshape((S,S,out))(x)
YOLO = Model(inputs = input, outputs = y)
YOLO.summary()

optimizer = optimizers.SGD(learing_rate,0.9,0.0005)
YOLO.compile(optimizer,loss=multipartloss)
train_batch = DataGenerator(train,batch_size)
val_batch = DataGenerator(val,batch_size)

YOLO.fit_generator(
	generator = train_batch,
	steps_per_epoch = len(train_batch),
	epochs = epochs,
	validation_data = val_batch,
	validation_steps = len(val_batch),
	callbacks=[LearningRateScheduler(lr_adaptive)],
	verbose = 1
)