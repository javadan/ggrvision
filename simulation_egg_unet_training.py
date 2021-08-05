
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import fnmatch
import re

import random
from datetime import datetime

from sklearn.model_selection import train_test_split


from keras_unet.utils import get_augmented
import tensorflow as tf
from PIL import Image
from collections import defaultdict

from keras_unet.models import custom_unet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded


train_images_dir = "jpgs/"
train_masks_dir = "pngs/"


#from keras_unet.losses import jaccard_distance

train_image_paths = sorted(
    [
        os.path.join(train_images_dir, fname)
        for fname in os.listdir(train_images_dir)
        if fname.endswith(".jpg")
    ]
)

train_mask_paths = sorted(
    [
        os.path.join(train_masks_dir, fname)
        for fname in os.listdir(train_masks_dir)
        if fname.endswith(".png")
    ]
)

print("Number of Train images:", len(train_image_paths))
print("Number of Train masks:", len(train_mask_paths))


imgs_list = []
masks_list = []

count = 0
for image in train_image_paths:
    count+=1    
    value = train_mask_paths[count-1]
    print(str(count - 1) + "    ./"+image+ "    ./"+value)

    
    i = Image.open(image).resize((256,256))
    i = i.convert('L') 
    imgs_list.append(np.array(i))

    m = Image.open(value).resize((256,256))
    m = m.convert('L') 
    masks_list.append(np.array(m))
        
imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

print(imgs_np.shape, masks_np.shape)
print(imgs_np.max(), masks_np.max())

x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255

print(x.max(), y.max())
print(x.shape, y.shape)
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape, y.shape)


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

train_gen = get_augmented(
    x_train, y_train, batch_size=4,
    data_gen_args = dict(
        horizontal_flip=True
    ))

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=32,
    dropout=0.2,
    output_activation='sigmoid'
)

# model.summary()

model.compile(
    optimizer=Adam(),
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)

try:
    model.load_weights(best_file_checkpoint)
    print("Loaded best weights:", best_file_checkpoint)
except Exception:
    print("Not loading weights")

fname = "sim_vision_weights-{val_loss:.4f}.hdf5"

callback_checkpoint = ModelCheckpoint(
    fname, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True
)
callbacks_list = [callback_checkpoint]


history = model.fit_generator(
    train_gen,
    steps_per_epoch=x_train.shape[0],
    epochs=10,

    validation_data=(x_val, y_val),
    callbacks=callbacks_list
)

