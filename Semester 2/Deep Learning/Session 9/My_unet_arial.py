# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 23:48:32 2021

@author: Senthil
"""
#Dataset Link:
# https://drive.google.com/file/d/1MU-vlQsvZL4ZgIjZhLPPj2tVC4gYGdXH/view?usp=sharing
# https://drive.google.com/file/d/1zVr-QafsT0itSsNq5kM426Zhrd7Pf8Db/view?usp=sharing
# https://drive.google.com/file/d/1PhWR38XMAm6yGkn7YY9FfOV21jz5lyQ1/view?usp=sharing
import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
#from patchify import patchify
from PIL import Image
import segmentation_models as sm
from tensorflow.keras.metrics import MeanIoU



image_dataset = np.load('images.npy')
mask_dataset =  np.load('masks.npy')
labels=np.load('labels.npy')

#plt.figure(1)
#plt.imshow(image_dataset[0,:,:,:])
#plt.figure(2)
#plt.imshow(mask_dataset[0,:,:,:])

n_classes = len(np.unique(labels))
from tensorflow.keras.utils import to_categorical
labels_cat = to_categorical(labels, num_classes=n_classes)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, labels_cat, test_size = 0.20, random_state = 42)


BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

# preprocess input
X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)


from tensorflow.keras import backend as K

def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

metrics=['accuracy', jacard_coef]

# define model
import segmentation_models as sm

BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)

X_train_prepr = preprocess_input(X_train)
X_test_prepr = preprocess_input(X_test)

model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=n_classes, activation='softmax')


# compile keras model with defined optimozer, loss and metrics
#model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
model_resnet_backbone.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

print(model_resnet_backbone.summary())


history2=model_resnet_backbone.fit(X_train_prepr, 
          y_train,
          batch_size=16, 
          epochs=50,
          verbose=1,
          validation_data=(X_test_prepr, y_test))

model_resnet_backbone.save('arial_unet_weights.hdf5')
#model.save('models/arial_unet_weights.hdf5.hdf5')

#from keras.models import load_model
#model = load_model("arial_unet_weights.hdf5",
#                   custom_objects={'dice_loss_plus_2focal_loss': total_loss,
#                                   'jacard_coef':jacard_coef})
    
model_resnet_backbone.load_weights('arial_unet_weights.hdf5')

y_pred=model_resnet_backbone.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
y_test_argmax=np.argmax(y_test, axis=3)


import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test_argmax[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model_resnet_backbone.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

plt.figure(1);plt.imshow(ground_truth)
plt.figure(2);plt.imshow(predicted_img)
plt.imshow(mask_dataset[0,:,:,:])