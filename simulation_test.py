!pip3 install keras-unet
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import glob
import os
import sys
import fnmatch
import re
import random
from datetime import datetime

import tensorflow as tf
from PIL import Image
from collections import defaultdict

from keras_unet.utils import get_augmented
from keras_unet.models import custom_unet
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance



test_images_dir = "../../../images/SimulationVision/cnn/"
test_masks_dir = "../../../images/SimulationVision/cnn/"

#test_images_dir = "jpgs/"
#test_masks_dir = "pngs/"


#from keras_unet.losses import jaccard_distance

test_image_paths = sorted(
    [
        os.path.join(test_images_dir, fname)
        for fname in os.listdir(test_images_dir)
        if fname.startswith("rgb") and fname.endswith(".jpg")
    ]
)

test_mask_paths = sorted(
    [
        os.path.join(test_masks_dir, fname)
        for fname in os.listdir(test_masks_dir)
        if fname.endswith(".png")
    ]
)


    
print("Number of test images:", len(test_image_paths))
print("Number of test masks:", len(test_mask_paths))



imgs_list = []
masks_list = []

random.seed(datetime.now())
enough = random.randrange(5, 2000)#len(test_image_paths) - 1)
startindex = enough - 5
counter = 0


imgs_list = []
masks_list = []

count = 0
for image in test_image_paths:
    count+=1    
    
    if count > startindex and count < enough:
        value = test_mask_paths[count-1]
        print(str(count - 1) + "    ./"+image+ "    ./"+value)


        i = Image.open(image).resize((256,256))
        i = i.convert('L') 
        imgs_list.append(np.array(i))

        m = Image.open(value).resize((256,256))
        #m = m.convert('L') 
        fullmask = np.array(m)

        #plane = (fullmask == 0)
        egg = (fullmask == 1)
        chicken = (fullmask == 2)
        robot = (fullmask == 3)
        human = (fullmask == 4)
        #sky = (fullmask == 255)

        masks = np.stack((egg, chicken, robot, human), axis=-1)

        masks_list.append(masks)

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)

print(imgs_np.shape, masks_np.shape)
print(imgs_np.max(), masks_np.max())
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)
print(x.max(), y.max())
print(x.shape, y.shape)

y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 4)
print(x.shape, y.shape)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
print(x.shape, y.shape)

for i, title in enumerate(['image']):
     plt.imshow(x[0][:,:,i])
     #plt.imshow(y[0][:,:,i])
     plt.title(title)
     plt.show()


for i, title in enumerate(['human', 'chicken', 'egg', 'robot']):
     plt.imshow(y[0][:,:,i])
     plt.title(title)
     plt.show()




train_gen = get_augmented(
    x, y, batch_size=4,
    data_gen_args = dict(
        horizontal_flip=True,
        zoom_range=0.3    ))


input_shape = x[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=True,
    num_classes=4,
    filters=32,
    dropout=0.2,
    output_activation='sigmoid'
)

model.summary()



sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("sim_vision_weights")])

#model_filename = 'chicken_training_varied.h5'

callback_checkpoint = ModelCheckpoint(
    sorted_weights[0], 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)


lr = random.uniform(0.001,0.0007)
model.compile(
    optimizer=Adam(learning_rate=lr), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)


model.load_weights(sorted_weights[0])
#y_pred = model.predict(x)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=np.stack((x[0], x[0], x[0], x[0])),
          mask_imgs=np.stack((y[0][:,:,0], y[0][:,:,1], y[0][:,:,2], y[0][:,:,3])),
          #pred_imgs=np.stack((y_pred[0][:,:,0], y_pred[0][:,:,1], y_pred[0][:,:,2], y_pred[0][:,:,3])),
          nm_img_to_plot=4)


