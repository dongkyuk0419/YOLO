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
import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from keras.layers import BatchNormalization, LeakyReLU, Lambda, Input
from keras.layers.merge import concatenate
from data_load import load_data
from keras.callbacks import LearningRateScheduler

# Data Load
ann_dir = '/home/dongkyu/VOC2012/Annotations/'
img_dir = '/home/dongkyu/VOC2012/JPEGImages/'
lab_dir = '/home/dongkyu/VOC2012/Labels/'
labels = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

train, val, test = load_data(ann_dir,img_dir,lab_dir,labels)

# Model Functions

def convconv(filters,size,input):
	x = Conv2D(filters,size,padding='same')(input)
	x = BatchNormalization()(x)
	x = LeakyReLU(0.1)(x)
	return x	

def convblock1(filters,input):
	x = convconv(filters,3,input)
	x = MaxPooling2D(strides=2)(x)
	return x

def convblock2(filters,input):
	x = convconv(filters*2,3,input)
	x = convconv(filters,1,x)
	x = convconv(filters*2,3,x)
	x = MaxPooling2D(strides=2)(x)
	return x

def convblock3(filters,input):
	x = convconv(filters*2,3,input)
	x = convconv(filters,1,x)
	x = convconv(filters*2,3,x)
	x = convconv(filters,1,x)
	x = convconv(filters*2,3,x)
	return x	

def reorg(x):
	return tf.space_to_depth(x,2)

def lr_adaptive(self,epoch):
    if epoch > 60:
        lr = 1e-4
    if epoch > 90:
        lr = 1e-5
    return lr

# Model
#
# We obtained details about the yolov2 structure on
# https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg
#

# Parameters
I_h = 416
S = 13
box = 5
classes = len(labels)
out = (5+classes)*box
learing_rate = 1e-3
epochs = 160

# YOLOv2 Model
input = Input(shape=(I_h,I_h,3))
x = convblock1(32,input)
x = convblock1(64,x)
x = convblock2(64,x)
x = convblock2(128,x)
x = convblock3(256,x)
passthrough = x
x = MaxPooling2D(strides=2)(x)
x = convblock3(512,x)
for i in range(0,2):
	x = convconv(1024,3,x)
passthrough = convconv(64,1,passthrough)
passthrough = Lambda(reorg)(passthrough)
x = concatenate([x,passthrough])
x = convconv(1024,3,x)
y = Conv2D(out,1,padding = 'same')(x)
YOLO = Model(inputs = input, outputs = y)
YOLO.summary()

def multiartloss(y,y_hat):





optimizer = optimizers.SGD(lr,0.9,0.0005)
#optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
YOLO.compile(optimizer,loss=multipartloss)
YOLO.fit
YOLO.fit_generator(epochs = epochs,
	callbacks=[LearningRateScheduler(lr_adaptive)]
	)




            self.optim = optimizers.Adam(learning_rate)
        self.model.compile(self.optim,'categorical_crossentropy',['accuracy'])
        self.datagen = keras.preprocessing.image.ImageDataGenerator(
            horizontal_flip = True,fill_mode = 'constant',
            width_shift_range = 4, height_shift_range = 4
            )
        self.datagen.fit(self.data.X_train)
        self.model.fit_generator(self.datagen.flow(self.data.X_train,self.data.Y_train,
            batch_size=self.batch_size),steps_per_epoch=len(self.data.X_train)/self.batch_size,
            epochs=self.epochs,validation_data = (self.data.X_val,self.data.Y_val),
            callbacks=[LearningRateScheduler(lr_adaptive)],verbose=2)
