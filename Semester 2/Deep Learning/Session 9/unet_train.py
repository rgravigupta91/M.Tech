# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 12:10:57 2021

@author: Senthil
"""
#Dataset Link:
#https://drive.google.com/drive/folders/1XEr6aXAVy41qV1HwRGAYkZk7KUmv-NPY?usp=sharing
#https://drive.google.com/drive/folders/1XhROpWIKKU8FZaMJaBdcr9Us-81JNFhs?usp=sharing


from simple_unet_model_with_jacard import simple_unet_model_with_jacard

from tensorflow.keras.utils import normalize
import os
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image_dir='C:\\Users\\Senthil\\Desktop\\mic\\images\\'

mask_dir='C:\\Users\\Senthil\\Desktop\\mic\\mask\\'

SIZE=256

img_dataset=[]
mask_dataset=[]

images=os.listdir(image_dir)

for i,image_name in enumerate(images):
    if (image_name.split('.')[1]=='tif'):
        image=cv2.imread(image_dir+image_name,0)
        image=Image.fromarray(image)
        image=image.resize((SIZE,SIZE))
        img_dataset.append(np.array(image))
        
        
masks=os.listdir(mask_dir)

for i,image_name in enumerate(masks):
    if (image_name.split('.')[1]=='tif'):
        image=cv2.imread(mask_dir+image_name,0)
        image=Image.fromarray(image)
        image=image.resize((SIZE,SIZE))
        mask_dataset.append(np.array(image))
        
img_dataset=np.array(img_dataset)
img_dataset=normalize(img_dataset)
img_dataset=np.expand_dims(img_dataset,3)

mask_dataset=np.array(mask_dataset)
mask_dataset=mask_dataset/255.
mask_dataset=np.expand_dims(mask_dataset,3)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(img_dataset,mask_dataset,test_size=0.10,random_state=0)

IMG_HEIGHT=img_dataset.shape[1]
IMG_WIDTH=img_dataset.shape[2]
IMG_CHANNELS=img_dataset.shape[3]

model=simple_unet_model_with_jacard(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS)

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=1,
          validation_data=(x_test,y_test),shuffle=False)
#model.save('mitochndria.hdf5)

model.load_weights('mitochondria_with_jacard.hdf5')

loss,jac_coef=model.evaluate(x_test,y_test)

n1=np.random.randint(0,len(x_test))
test_img=x_test[n1]
mask_test_img=y_test[n1]
test_img1=np.expand_dims(test_img,0)
pred_img=model.predict(test_img1)
pred_img1=(pred_img[0,:,:,0]>0.5).astype(np.uint8)

plt.figure(figsize=(16,8))
plt.subplot(131)
plt.title('Original')
plt.imshow(test_img[:,:,0],cmap='gray')

plt.subplot(132)
plt.title('Mask Original')
plt.imshow(mask_test_img[:,:,0],cmap='gray')

plt.subplot(133)
plt.title('Segmented Image')
plt.imshow(pred_img1,cmap='gray')
