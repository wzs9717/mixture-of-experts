from __future__ import print_function
import keras
import sys
sys.setrecursionlimit(10000)

#import densenet
import numpy as np
import sklearn.metrics as metrics
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from keras.preprocessing.image import img_to_array, array_to_img
from keras.models import Sequential,load_model
from PIL import Image
import tensorflow as tf
import gc
import pandas as pd
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
import os
from keras.utils import multi_gpu_model
from keras import backend as K
import matplotlib.pyplot as plt
import cv2
from scipy import io
import struct
def load_images(file_name):
    binfile = open(file_name, 'rb') 
    buffers = binfile.read()
    magic,num,rows,cols = struct.unpack_from('>IIII',buffers, 0)
    bits = num * rows * cols
    images = struct.unpack_from('>' + str(bits) + 'B', buffers, struct.calcsize('>IIII'))
    binfile.close()
    images = np.reshape(images, [num, rows * cols])
    return images

def pic_transfer(X):
#    resize X from 32 by 32 to 224by224
    # X_resize=np.zeros((224,224,3))
    pic_tem=array_to_img(X)
    pic_tem2=pic_tem.resize((224, 224),Image.BILINEAR)
    X_resize=img_to_array(pic_tem2)
    return X_resize

def main():

    model=load_model('./weights/MobilenetV2-CIFAR100.h5')

    data = load_images('./digital/t10k-images-idx3-ubyte')
    # print(data.keys())
    # images=data
    data.shape=[data.shape[0],28,28]

    # images=cv2.imread('/home/kleong013/Documents/intern document/wang zhisheng/detection/MobileNet-master (2)/model_3/data/test/0.jpg')
    # Turn the image into an array.
    # image_arr = np.expand_dims(images, axis=0)
 
# feature extraction
    layer_1 = K.function([model.layers[0].input], [model.get_layer('global_average_pooling2d_1').output])
    features=np.zeros((data.shape[0],1024))
    data2=np.zeros((28,28,3))
    for i in range(data.shape[0]):
        data2[:,:,0]=data[i,:,:]
        data2[:,:,1]=data[i,:,:]
        data2[:,:,2]=data[i,:,:]
        tem=pic_transfer(data2.reshape((28,28,3)))
        #data_tem[:,:,0]=tem.reshape((224,224))
        #data_tem[:,:,1]=tem.reshape((224,224))
        #data_tem[:,:,2]=tem.reshape((224,224))
        tem = np.expand_dims(tem,axis=0)
        f1 = layer_1([tem])[0]
        #print(f1.shape)
        f1.shape = [1,1024]#need to change
        features[i,:]=f1
        if i%500==0:
            print(i)
            print(f1)
    io.savemat('data_feature_test.mat', {'test': features})
    # conv layer: 299

    # layer_name='block_1_depthwise'
    # intermediate_layer_model = Model(input=model.input,
    #                              output=model.get_layer(layer_name).output)
    print('finished !')
 
if __name__ == '__main__':
    main()

