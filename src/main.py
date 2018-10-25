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
# The pretraining portion of the paper was skipped, because it required
# imagenet, gtx-titan and few weeks.
#
# We present some results of the project.
# For some reason our model thinks everything is a person.
# We believe this is due to the model's error not converging.
# We tried training the model with different learning rate, batch size,
# over few decades of epochs.

# Library and Function Imports
import tensorflow as tf 
import numpy as np
import keras
import cv2
from keras import optimizers
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten, Reshape
from keras.layers import BatchNormalization, LeakyReLU, Lambda, Input, Dropout
from keras.layers.merge import concatenate
from keras.callbacks import LearningRateScheduler
from data_load import load_data, DataGenerator
from matplotlib import pyplot as plt
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

def overlap(x1, w1, x2, w2):
    l1 = x1-w1/ 2.
    l2 = x2-w2/ 2.
    left = tf.maximum(l1, l2)
    r1 = x1 + w1 / 2.
    r2 = x2 + w2 / 2.
    right = tf.minimum(r1, r2)
    return right - left

class Box(object):
	def __init__(self):
		self.x = float()
		self.y = float()
		self.w = float()
		self.h = float()
		self.c = float()
		self.p = float()

def multipartloss(y,y_hat):
	coord = 5.
	noobj = 0.5
	l= coord*np.multiply(tf.reduce_sum(np.power((y_hat[:,:,:,0:4]-y[:,:,:,0:4]),2),[3]),y[:,:,:,5])
	l+= coord*np.multiply(tf.reduce_sum(np.power((y_hat[:,:,:,5:9]-y[:,:,:,0:4]),2),[3]),y[:,:,:,5])
	l+=np.multiply(tf.reduce_sum(np.power((y_hat[:,:,:,10:30]-y[:,:,:,10:30]),2),3),y[:,:,:,5])
	
	yx = y[:,:,:,0]
	yy = y[:,:,:,1]
	yw = y[:,:,:,2]
	yh = y[:,:,:,3]
	yhatx = y_hat[:,:,:,0]
	yhaty = y_hat[:,:,:,1]
	yhatw = y_hat[:,:,:,2]
	yhath = y_hat[:,:,:,3]
	ww = tf.maximum(overlap(yx,yw,yhatx,yhatw),0)
	hh = tf.maximum(overlap(yy,yh,yhaty,yhath),0)
	intersec = ww*hh
	IOU = intersec / (yw*yh+yhatw*yhath - intersec)
	C = IOU*tf.reduce_max(y_hat[:,:,:,10:30]*y[:,:,:,10:30],3)
	l+=tf.reduce_sum(np.multiply(np.power(C-y[:,:,:,4],2),y[:,:,:,5]))
	l+=noobj*tf.reduce_sum(np.multiply(np.power(C-y[:,:,:,4],2),y[:,:,:,6]))
	
	return l


def boxes_form(out,threshold,S,B,C):
	boxes = []
	boxes_labels = []
	box_info = out[:,:,:,0:10].reshape((1,7,7,2,-1))
	category = out[:,:,:,10:30]
	for grid_x in range(S):
		for grid_y in range(S):
			for bb in range(B):
				box = Box()
				box.x = (box_info[0,grid_x,grid_y,bb,0]+grid_x)/S*448
				box.y = (box_info[0,grid_x,grid_y,bb,1]+grid_y)/S*448
				box.w = box_info[0,grid_x,grid_y,bb,2]**2*448
				box.h = box_info[0,grid_x,grid_y,bb,3]**2*448
				box.c = box_info[0,grid_x,grid_y,bb,4]
				box.p = category[0,grid_x,grid_y,:]*box.c
				k = np.max(box.p)
				print(box.p)

				for i in range(C):
					if (box.p[i] >=threshold) and box.p[i] == k and box.w > 10 and box.h >10:
						boxes.append(box)
						boxes_labels.append(labels[i])
	#print(boxes)
	print(boxes_labels)
	return boxes, boxes_labels

def draw_boxes(sample_img,boxes,boxes_labels):
	imgcp = sample_img.copy()
	for ii in range(len(boxes)):
		i = boxes[ii]
		l = int((i.x-i.w/2))
		r = int((i.x+i.w/2))
		t = int((i.y+i.h/2))
		b  = int((i.y-i.h/2))
		if l < 0:
			l = 0
		if r > 447:
			r = 447
		if t > 447:
			t = 447
		if b < 0:
			b = 0
		imgcp = cv2.rectangle(imgcp,(l,b),(r,t),(255,0,0),2)
		imgcp = cv2.putText(imgcp,boxes_labels[ii],(l,b),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),lineType=cv2.LINE_AA)
	return imgcp
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
learning_rate = 1e-4
epochs = 10
batch_size = 16
threshold = 0.85 # The threshold is really sensitive!! Something to fix.
def lr_adaptive(self,epoch):
    lr = learning_rate
    if epoch > 20:
    	lr = learning_rate/10
    if epoch > 30:
        lr = learning_rate/100
    if epoch > 50:
        lr = learning_rate/1000
    return lr

# YOLO Model_tiny
# The original YOLO model was too large, and took forever to train.
# We switched to a much affordable tiny YOLO.
# We also decreased the batch_size from 64 to 16, because it crashed my computer.
input2 = Input(shape=(I_h,I_h,3))
x2 = convblock1(16,3,1,input2)
x2 = convblock1(32,3,1,x2)
x2 = convblock1(64,3,1,x2)
x2 = convblock1(128,3,1,x2)
x2 = convblock1(256,3,1,x2)
x2 = convblock1(512,3,1,x2)
x2 = convconv(1024,3,1,x2)
x2 = convconv(1024,3,1,x2)
x2 = convconv(1024,3,1,x2)
x2 = Flatten()(x2)
x2 = Dense(256)(x2)
x2 = Dense(4096)(x2)
x2 = LeakyReLU(0.1)(x2)
x2 = Dense(S*S*out)(x2)
y2 = Reshape((S,S,out))(x2)
model = Model(inputs = input2, outputs = y2)
#model.summary()
model.load_weights('weights3.h5')

# Training Loop
def traintrain():
	optimizer = optimizers.SGD(learning_rate,0.9,0.0005)
	model.compile(optimizer,loss=multipartloss)
	train_batch = DataGenerator(train,batch_size)
	val_batch = DataGenerator(val,batch_size)

	model.fit_generator(
		generator = train_batch,
		steps_per_epoch = len(train_batch),
		epochs = epochs,
		validation_data = val_batch,
		validation_steps = 10,#len(val_batch),
		callbacks=[LearningRateScheduler(lr_adaptive)],
	)
	model.save_weights('weights4.h5')
	return 0

# Test Loop
def testtest():
	# Figure 
	#sample_img = cv2.resize(cv2.imread(img_dir+'2008_004504.jpg'),(448,448))/255
	#sample_img = cv2.resize(cv2.imread(img_dir+'2011_004117.jpg'),(448,448))/255
	#sample_img = cv2.resize(cv2.imread('./sample_img.jpg'),(448,448))/255
	sample_img = cv2.resize(cv2.imread(test[500]['file']),(448,448))/255


	sample_out = model.predict(np.expand_dims(sample_img,axis=0))
	boxes, boxes_labels = boxes_form(sample_out,threshold,S,B,C)
	box_drawn = draw_boxes(sample_img,boxes,boxes_labels)

	plt.subplot(1,2,1)
	plt.imshow(sample_img)
	plt.axis('off')
	plt.title("Original")
	plt.subplot(1,2,2)
	plt.imshow(box_drawn)
	plt.axis('off')
	plt.title("Box")
	plt.show()

# Execution

#traintrain()
testtest()