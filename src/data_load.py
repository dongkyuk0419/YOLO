# The paper mentions that the YOLO model is pretrained using ImageNet for a week
# using ImageNet, and we decided to skip this section because the reference
# the paper gives does not contain any information about how they did the 
# (pre)-training.
#
# About the data:
# The data was downloaded from:
# http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# The following processing was done:
# 1. File was unzipped to get VOC2012 folder.
# 2. In the ImageSets folder, anything other than Main folder was deleted and
#   the Main folder was renamed as Labels and moved to the VOC2012.
# 3. Because the detection task does not consider 2007 and 2012 data, all the
#   jpegs in JPEGImages that correspond to 2007 were deleted, as well as
#   annotations in Annotations folder.
# 4. SegmentationClass and SegmentationObject folders were completely deleted.
# 
# The data was stored in: /home/dongkyu/VOC2012

import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from keras.utils import Sequence

def load_data(ann_dir,img_dir,lab_dir,labels):
    train_index = []
    val_index = []
    k = open(lab_dir+'train.txt')
    kk = open(lab_dir+'val.txt')
    for line in k.readlines():
        train_index += line.split()
    for line in kk.readlines():
        val_index += line.split()

    train_infos = []
    val_infos = []
    test_infos = []

    for ann in sorted(os.listdir(ann_dir)):
        infos_temp = {'object':[]}
        tree = ET.parse(ann_dir + ann)
        for elem in tree.iter():
            if elem.tag == 'filename':
                infos_temp['file'] = img_dir + elem.text
                token = 0
                if elem.text[0:11] in train_index:
                    token = 1
                if elem.text[0:11] in val_index:
                    token = 2
            if elem.tag == 'width':
                infos_temp['width'] = int(elem.text)
            if elem.tag == 'height':
                infos_temp['height'] = int(elem.text)
            if elem.tag == 'depth':
                infos_temp['depth'] = int(elem.text)
            if elem.tag == 'object':
                object_info = {}                
                for elem2 in elem:
                    if elem2.tag == 'name':
                        if elem2.text in labels:
                            object_info['name'] = elem2.text
                        else:
                            print('error')
                            break
                    if elem2.tag == 'bndbox':
                        for xy in elem2:
                            if xy.tag == 'xmin':
                                object_info['xmin'] = int(float(xy.text))
                            if xy.tag == 'xmax':
                                object_info['xmax'] = int(float(xy.text))
                            if xy.tag == 'ymin':
                                object_info['ymin'] = int(float(xy.text))
                            # some values of ymin are float for some reaon
                            if xy.tag == 'ymax':
                                object_info['ymax'] = int(float(xy.text))
                infos_temp['object'] += [object_info]
        if len(infos_temp['object']) > 0:
            if token == 1:
                train_infos += [infos_temp]
            if token == 2:
                if np.random.rand() > 0.9:
                    test_infos += [infos_temp]
                else:
                    val_infos += [infos_temp]
    return train_infos, val_infos, test_infos

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class DataGenerator(Sequence):
    def __init__(self, infos,batch_size):
        self.infos = infos
        self.batch_size = batch_size
        self.objects = ['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    
    def __len__(self):
        return int(np.ceil(len(self.infos)/self.batch_size))

    def __getitem__(self,index):
        start = index*self.batch_size
        end = (index+1)*self.batch_size
        if end > len(self.infos):
            end = len(self.infos)
            start = end - self.batch_size
        x = np.zeros((self.batch_size,448,448,3))
        y = np.zeros((self.batch_size,7,7,30))
        y[:,:,:,6] = 1
        counter = 0
        #preprocessing
        for img in self.infos[start:end]:
            x[counter] = cv2.resize(cv2.imread(img['file']),(448,448))/255
            h_org = img['height']
            w_org = img['width']
            x_center = []
            y_center = []
            width = []
            height = []
            p_cond = []
            for obj in img['object']:
                x_start = obj['xmin']/w_org*448
                x_end = obj['xmax']/w_org*448
                y_start = obj['ymin']/h_org*448
                y_end = obj['ymax']/h_org*448
                x_temp = int((x_start+x_end)/2)
                y_temp = int((y_start+y_end)/2)
                w_temp = int(x_end - x_start)/448
                h_temp = int(y_end - y_start)/448
                grid_x = int(np.floor(x_temp/64.))
                grid_y = int(np.floor(y_temp/64.))
                y[counter,grid_x,grid_y,0:5] = [(x_temp%64)/64,(y_temp%64)/64,np.sqrt(w_temp),np.sqrt(h_temp),1]
                y[counter,grid_x,grid_y,5] = 1
                y[counter,grid_x,grid_y,6] = 0
                for index in range(len(self.objects)):
                    if obj['name'] == self.objects[index]:
                        y[counter,grid_x,grid_y,10+index] =1
            counter += 1
        return x,y
        
    def on_epoch_end(self):
        np.random.shuffle(self.infos)