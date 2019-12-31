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
from keras.models import Sequential
from PIL import Image
import tensorflow as tf
import gc
import pandas as pd
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, Add, Reshape, DepthwiseConv2D
import os
from keras.utils import multi_gpu_model


def main():

  class para_ModelCheckpoint(ModelCheckpoint):
      def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                   save_best_only=False, save_weights_only=False,
                   mode='auto', period=1):
          self.single_model = model
          super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

      def set_model(self, model):
          super(ParallelModelCheckpoint,self).set_model(self.single_model)

  os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
  
  print(os.environ['CUDA_VISIBLE_DEVICES'])

  def pic_transfer(X):
  #    resize X from 32 by 32 to 224by224
      number_X=10
      X_resize=np.zeros((number_X,224,224,3))
      for i in range(number_X):
          pic_tem=array_to_img(X[i,:,:,:])
          pic_tem2=pic_tem.resize((224, 224),Image.BILINEAR)
          X_resize[i,:,:,:]=img_to_array(pic_tem2)
      return X_resize
      
      
      
  batch_size = 128
  nb_classes = 100
  nb_epoch = 6000
  classes=100
  img_rows, img_cols = 224, 224
  img_channels = 3

  img_dim = (img_rows, img_cols, img_channels)
  depth = 40
  nb_dense_block = 3
  growth_rate = 12
  nb_filter = 12
  bottleneck = False
  reduction = 0.0
  dropout_rate = 0.0 # 0.0 for data augmentation

  #--------------------------------------------------------------------------------------------
  alpha=1.0
  #pre_model = Sequential()

  base_model=keras.applications.mobilenet.MobileNet(input_shape=img_dim, alpha=alpha, include_top=False, weights="imagenet", input_tensor=None, pooling=None, classes=classes)
  shape = (1, 1, int(1280 * alpha))
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  #x = keras.layers.Flatten(name='flatten')(x)
  x = keras.layers.Dense(classes, activation='softmax',
                         use_bias=True, name='Logits')(x)
  # x = GlobalAveragePooling2D()(x)
  # x = Activation('softmax', name='softmax')(x)
  # output = Reshape((nb_classes,))(x)
  inputs = base_model.input
  model = keras.models.Model(inputs, x)

  for layer in model.layers[0:-1]:
        layer.trainable = False
  for layer in model.layers[-30:-1]:
         layer.trainable = True
  #model = multi_gpu_model(model, gpus=2)
  print("Model created")
  #--------------------------------------------------------------------------------------------
  model.summary()
  optimizer = Adam(lr=1e-4) # Using Adam instead of SGD to speed up training
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])
  print("Finished compiling")
  print("Building model...")

  (trainX, trainY), (testX, testY) = cifar100.load_data()
  trainY = trainY.reshape(trainY.shape[0])
  testY = testY.reshape(testY.shape[0])
  print(trainX.shape,trainY.shape)
  print(testX.shape,testY.shape)
  trainX = trainX.astype('float32')
  testX = testX.astype('float32')
  Y_train = np_utils.to_categorical(trainY, nb_classes)
  Y_test = np_utils.to_categorical(testY, nb_classes)
  print(Y_train.shape,Y_test.shape)
  # trainX=pic_transfer(trainX)
  # testX=pic_transfer(testX)
  # trainY=trainY[0:10]
  # testY=testY[0:10]

  #del trainX, trainYtestX, testY
  #gc.collect()

  def append_ext(fn):
      return str(fn)+".jpg"

  def append_ext2(fn):
      return str(fn)

  def transferY(Y):
      Y_new=[]
      for i in range(Y.shape[0]):
        Y_new.append(np.matrix.tolist(Y[i,:]))
      return Y_new


  traindf=pd.DataFrame({'id':np.arange(trainY.shape[0]),'label':(trainY)},index=np.arange(trainY.shape[0]))
  testdf=pd.DataFrame({'id':np.arange(testY.shape[0]),'label':(testY)},index=np.arange(testY.shape[0]))

  traindf["id"]=traindf["id"].apply(append_ext)
  testdf["id"]=testdf["id"].apply(append_ext)

  traindf['label']=traindf['label'].apply(append_ext2)
  testdf['label']=testdf['label'].apply(append_ext2)

  print(testdf)
  datagen=ImageDataGenerator(rotation_range=15,
                                  width_shift_range=5./224,
                                  height_shift_range=5./224,
  				rescale=1./255.)

  train_generator = datagen.flow_from_dataframe(dataframe = traindf,
                                            directory="./data/train/",
                                            x_col="id",
                                            y_col="label",
                                            subset=None,
                                            batch_size=2*batch_size,
                                            seed=nb_epoch,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(224,224))

  valid_generator = datagen.flow_from_dataframe(dataframe = testdf,
                                            directory="./data/test/",
                                            x_col="id",
                                            y_col="label",
                                            subset=None,
                                            batch_size=batch_size,
                                            seed=nb_epoch,
                                            shuffle=True,
                                            class_mode="categorical",
                                            target_size=(224,224))

  test_datagen = ImageDataGenerator(rescale = 1. / 255.) 
  test_generator = test_datagen.flow_from_dataframe(dataframe = testdf,
                                            directory="./data/test/",
                                            x_col="id",
                                            y_col=None,
                                            
                                            batch_size=batch_size,
                                            seed=nb_epoch,
                                            shuffle=False,
                                            class_mode=None,
                                            target_size=(224,224))




  # trainX /= 255.
  # testX /= 255.


  #print(Y_train.shape)
  #print(Y_test.shape)
  # generator = ImageDataGenerator(rotation_range=15,
  #                                width_shift_range=5./32,
  #                                height_shift_range=5./32,
  #                               )

  # )
  # generator.fit(trainX, seed=0)
  #generator.fit(testX, seed=0)

  #transform_parameters['zx']=7

  #testX=apply_transform(testX, zx=7,zy=7)

  # Load model
  model.load_weights("weights/MobilenetV2-CIFAR100.h5")
  print("Model loaded.")

  lr_reducer      = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1),
                                      cooldown=0, patience=10, min_lr=0.5e-6)
  early_stopper   = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=20)
  model_checkpoint= ModelCheckpoint("weights/MobilenetV2-CIFAR100.h5", monitor="val_acc", save_best_only=True,
                                    save_weights_only=False)

  callbacks=[lr_reducer, early_stopper, model_checkpoint]


  #model.fit_generator(generator.flow(trainX, Y_train, batch_size=batch_size), samples_per_epoch=len(trainX), nb_epoch=nb_epoch,
   #                   callbacks=callbacks,
   #				   validation_data=(testX, Y_test),
   #                   nb_val_samples=testX.shape[0], verbose=1)

  STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
  STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
  STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

  model.fit_generator(generator=train_generator,
                      steps_per_epoch=STEP_SIZE_TRAIN,
                      callbacks=callbacks,
                      validation_data=valid_generator,
                      validation_steps=STEP_SIZE_VALID,
                      epochs=nb_epoch
                     )

  model.evaluate_generator(generator=valid_generator,
  steps=STEP_SIZE_TEST)

  # test_generator.reset()
  # pred=model.predict_generator(test_generator,
  # steps=STEP_SIZE_TEST,
  # verbose=1)

  # predicted_class_indices=np.argmax(pred,axis=1)

  # labels = (train_generator.class_indices)
  # labels = dict((v,k) for k,v in labels.items())
  # predictions = [labels[k] for k in predicted_class_indices]


  # filenames=test_generator.filenames
  # results=pd.DataFrame({"Filename":filenames,
  #                       "Predictions":predictions})
  # results.to_csv("results.csv",index=False)
  print(str(i)+'finished!')
  #yPreds = model.predict(testX)
  #yPred = np.argmax(yPreds, axis=1)
  #yTrue = testY

  #accuracy = metrics.accuracy_score(yTrue, yPred) * 100
  #error = 100 - accuracy
  #print("Accuracy : ", accuracy)
  #print("Error : ", error)

if __name__ == '__main__':
  for i in range(2):
    print(str(i)+"batch!!!!")
    main()
