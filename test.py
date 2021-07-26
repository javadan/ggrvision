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

base_dir = "../../../"

test_images_dir = base_dir + "images/Chicken/test/"
test_masks_dir = base_dir + "masks/Chicken/test/"
masks_dir = base_dir + "masks/"

test_image_paths = sorted(
    [
        os.path.join(test_images_dir, fname)
        for fname in os.listdir(test_images_dir)
        if fname.endswith(".jpg")
    ]
)

test_mask_paths = sorted(
    [
        os.path.join(test_masks_dir, fname)
        for fname in os.listdir(test_masks_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

    
print("Number of test images:", len(test_image_paths))
print("Number of test masks:", len(test_mask_paths))


for fname in os.listdir(test_images_dir):
    
    #print( test_masks_dir + "*" + fname[:-4] + "*" )
    if len(glob.glob(test_masks_dir + "*" + fname[:-4] + "*")) == 0:
        print(fname)
    
import tensorflow as tf
from PIL import Image
from collections import defaultdict

test_map = defaultdict(list)

for fname in os.listdir(test_images_dir):
    for mask in glob.glob(test_masks_dir + "*" + fname[:-4] + "*"):
        test_map[fname].append(mask)


imgs_list = []
masks_list = []

random.seed(datetime.now())
enough = random.randrange(5, len(test_image_paths) - 1)
startindex = enough - 5
counter = 0

for mask in test_map:
    counter += 1
    
    if (counter > startindex and counter <= enough):
    
        print(mask)
        print(test_map[mask])

        # Display auto-contrast version of corresponding target (per-pixel categories)
        for value in test_map[mask]:
            i = Image.open(test_images_dir + mask).resize((256,256))
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

from keras_unet.utils import get_augmented

train_gen = get_augmented(
    x, y, batch_size=1,
    data_gen_args = dict(    ))

from keras_unet.models import custom_unet

input_shape = x[0].shape

model = custom_unet(
    input_shape,
    use_batch_norm=False,
    num_classes=1,
    filters=32,
    dropout=0.2,
    output_activation='sigmoid'
)

model.summary()

from keras.callbacks import ModelCheckpoint


#model_filename = 'chicken_training_varied.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)

from keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

model.compile(
    optimizer=Adam(), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)


sorted_weights = sorted([fname for fname in os.listdir('.') if fname.startswith("weight")])

print(sorted_weights[0])
model.load_weights(sorted_weights[0])
y_pred = model.predict(x)

from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=x, mask_imgs=y, pred_imgs=y_pred, nm_img_to_plot=9)
